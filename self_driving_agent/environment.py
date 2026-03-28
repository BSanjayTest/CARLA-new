import glob
import os
import sys
import numpy as np
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import pickle
import math

from synch_mode import CarlaSyncMode
from controllers import PIDLongitudinalController
from utils import *

random.seed(78)


class SimEnv(object):
    def __init__(self,
        visuals=True,
        target_speed=30,
        max_iter=4000,
        start_buffer=10,
        train_freq=1,
        save_freq=200,
        start_ep=0,
        max_dist_from_waypoint=20,
        start_point_index=None,
        goal_point_index=None,
    ) -> None:
        self.visuals = visuals
        if self.visuals:
            self._initiate_visuals()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.load_world('Town02_Opt')
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)

        self.spawn_points = self.world.get_map().get_spawn_points()

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.find('vehicle.nissan.patrol')

        self.global_t = 0
        self.target_speed = target_speed
        self.max_iter = max_iter
        self.start_buffer = start_buffer
        self.train_freq = train_freq
        self.save_freq = save_freq
        self.start_ep = start_ep
        self.max_dist_from_waypoint = max_dist_from_waypoint
        self.start_train = self.start_ep + self.start_buffer

        self.total_rewards = 0
        self.average_rewards_list = []

        # Start/Goal routing
        self.start_point_index = start_point_index
        self.goal_point_index = goal_point_index
        self.route = None
        self.route_waypoints = []
        self.current_route_index = 0

        # Camera mode: 1=Front, 2=Third-Person, 3=Top-Down, 4=Side, 5=Semantic, 6=Depth, 7=Lidar, 8=Radar
        self.current_camera_mode = 2

    def _initiate_visuals(self):
        pygame.init()
        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()

    # ── Route Planning ──────────────────────────────────────────────
    def _generate_route(self):
        if self.goal_point_index is None:
            return
        try:
            # Find CARLA agents path (try env var, then common locations)
            _carla_root = os.environ.get('CARLA_ROOT', '')
            _carla_agents = os.path.join(_carla_root, 'PythonAPI', 'carla') if _carla_root else ''
            if not _carla_agents or not os.path.exists(_carla_agents):
                # Search common install locations
                for _drive in ['C', 'D', 'E']:
                    for _folder in ['CARLA', 'CARLA_0.9.16', 'selfdriving', 'carla']:
                        _test = f'{_drive}:\\{_folder}\\PythonAPI\\carla'
                        if os.path.exists(_test):
                            _carla_agents = _test
                            break
                    if _carla_agents:
                        break
            if _carla_agents and _carla_agents not in sys.path:
                sys.path.append(_carla_agents)
            from agents.navigation.global_route_planner import GlobalRoutePlanner
        except ImportError as e:
            print(f"WARNING: Route planning unavailable: {e}")
            return

        start_transform = self.spawn_points[self.start_point_index]
        goal_transform = self.spawn_points[self.goal_point_index]

        sampling_resolution = 2.0
        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution)

        # Round trip: start == goal, go to far point and come back
        if self.start_point_index == self.goal_point_index:
            # Pick the farthest spawn point as the turnaround
            max_dist = 0
            farthest_idx = 0
            for i, sp in enumerate(self.spawn_points):
                if i == self.start_point_index:
                    continue
                d = start_transform.location.distance(sp.location)
                if d > max_dist:
                    max_dist = d
                    farthest_idx = i

            mid_transform = self.spawn_points[farthest_idx]
            leg_out = grp.trace_route(start_transform.location, mid_transform.location)
            leg_back = grp.trace_route(mid_transform.location, start_transform.location)

            self.route = list(leg_out) + list(leg_back)
            self.route_waypoints = [wp for (wp, _) in self.route]
            self.current_route_index = 0

            print(f"Round-trip route: {len(self.route_waypoints)} waypoints")
            print(f"Start/Goal: point {self.start_point_index} ({start_transform.location.x:.1f}, {start_transform.location.y:.1f})")
            print(f"Turnaround: point {farthest_idx} ({mid_transform.location.x:.1f}, {mid_transform.location.y:.1f})")
        else:
            self.route = grp.trace_route(start_transform.location, goal_transform.location)
            self.route_waypoints = [wp for (wp, _) in self.route]
            self.current_route_index = 0

            print(f"Route generated: {len(self.route_waypoints)} waypoints")
            print(f"Start: point {self.start_point_index} ({start_transform.location.x:.1f}, {start_transform.location.y:.1f})")
            print(f"Goal:  point {self.goal_point_index} ({goal_transform.location.x:.1f}, {goal_transform.location.y:.1f})")

    def _get_next_route_waypoint(self):
        if not self.route_waypoints:
            return None
        vehicle_loc = self.vehicle.get_location()
        min_dist = float('inf')
        best_idx = self.current_route_index
        search_end = min(self.current_route_index + 20, len(self.route_waypoints))
        for i in range(self.current_route_index, search_end):
            dist = vehicle_loc.distance(self.route_waypoints[i].transform.location)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        self.current_route_index = best_idx
        return self.route_waypoints[best_idx]

    def _is_goal_reached(self):
        if not self.route_waypoints or self.goal_point_index is None:
            return False
        goal_loc = self.spawn_points[self.goal_point_index].location
        dist = self.vehicle.get_location().distance(goal_loc)
        # Only count as reached if very close to goal AND near end of route
        near_end = self.current_route_index >= len(self.route_waypoints) - 2
        return dist < 5.0 and near_end

    def _get_route_progress(self):
        if not self.route_waypoints:
            return 0.0
        return self.current_route_index / max(len(self.route_waypoints) - 1, 1)

    def create_actors(self):
        self.actor_list = []

        # Generate route if start/goal specified
        if self.start_point_index is not None and self.goal_point_index is not None:
            self._generate_route()

        # Spawn vehicle
        if self.start_point_index is not None:
            self.vehicle = self.world.try_spawn_actor(
                self.vehicle_blueprint, self.spawn_points[self.start_point_index])
            if self.vehicle is None:
                print(f"Failed to spawn at point {self.start_point_index}, using random")
                self.vehicle = self.world.spawn_actor(
                    self.vehicle_blueprint, random.choice(self.spawn_points))
            else:
                print(f"Spawned at start point {self.start_point_index}")
        else:
            self.vehicle = self.world.spawn_actor(
                self.vehicle_blueprint, random.choice(self.spawn_points))
        self.actor_list.append(self.vehicle)

        # Front camera (AI view)
        self.camera_rgb = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb)

        # Third-person camera
        self.camera_rgb_vis = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb_vis)

        # Top-down camera
        self.camera_top = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=0, z=12), carla.Rotation(pitch=-90)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_top)

        # Side camera
        self.camera_side = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=0, y=-10, z=2), carla.Rotation(yaw=-90)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_side)

        # Semantic segmentation camera
        self.camera_sem = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_sem)

        # Depth camera
        self.camera_depth = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.depth'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_depth)

        # Lidar sensor
        self.lidar = self.world.spawn_actor(
            self.blueprint_library.find('sensor.lidar.ray_cast'),
            carla.Transform(carla.Location(x=0, z=2.4)),
            attach_to=self.vehicle)
        self.actor_list.append(self.lidar)

        # Radar sensor
        self.radar = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.radar'),
            carla.Transform(carla.Location(x=2.8, z=1.0), carla.Rotation(pitch=5)),
            attach_to=self.vehicle)
        self.actor_list.append(self.radar)

        # Collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)

        # GNSS sensor
        self.gnss = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.gnss'),
            carla.Transform(carla.Location(x=1.0, z=2.4)),
            attach_to=self.vehicle)
        self.actor_list.append(self.gnss)

        # IMU sensor
        self.imu = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.imu'),
            carla.Transform(carla.Location(x=0, z=2.4)),
            attach_to=self.vehicle)
        self.actor_list.append(self.imu)

        self.speed_controller = PIDLongitudinalController(self.vehicle)

    # ── Reset / Cleanup ─────────────────────────────────────────────
    def reset(self):
        # Stop all sensors first to prevent socket errors
        for actor in self.actor_list:
            try:
                if actor.is_alive and 'sensor' in actor.type_id:
                    actor.stop()
            except:
                pass
        # Batch destroy all actors at once for instant respawn
        actor_ids = [actor.id for actor in self.actor_list]
        try:
            self.client.apply_batch([carla.command.DestroyActor(aid) for aid in actor_ids])
        except:
            for actor in self.actor_list:
                try:
                    if actor.is_alive:
                        actor.destroy()
                except:
                    pass
        self.actor_list = []
        self.world.tick()  # Flush the destruction before spawning new actors

    def quit(self):
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except:
            pass
        try:
            for actor in self.world.get_actors():
                if any(x in actor.type_id for x in ['vehicle', 'sensor', 'walker', 'controller']):
                    try:
                        if actor.is_alive:
                            actor.destroy()
                    except:
                        pass
        except:
            pass
        try:
            self.client.load_world('Town02_Opt')
            print("Simulator reset to fresh state.")
        except:
            pass
        try:
            pygame.quit()
        except:
            pass

    # ── Episode Generation ──────────────────────────────────────────
    def generate_episode(self, model, replay_buffer, ep, action_map=None, eval=True):
        sensors = [
            self.camera_rgb,
            self.camera_rgb_vis,
            self.camera_top,
            self.camera_side,
            self.camera_sem,
            self.camera_depth,
            self.lidar,
            self.radar,
            self.gnss,
            self.imu,
        ]

        with CarlaSyncMode(self.world, *sensors, self.collision_sensor, fps=30) as sync_mode:
            counter = 0

            tick_data = sync_mode.tick(timeout=0.5)
            if not tick_data or tick_data[0] is None:
                print("No data, skipping episode")
                self.reset()
                return None

            snapshot, image_rgb, image_rgb_vis, image_top, image_side, \
                image_sem, image_depth, lidar_data, radar_data, gnss_data, imu_data, collision = tick_data

            if image_rgb is None:
                print("No image data, skipping episode")
                self.reset()
                return None

            # Initial waypoint
            vehicle_location = self.vehicle.get_location()
            if self.route_waypoints:
                route_wp = self._get_next_route_waypoint()
                waypoint = route_wp if route_wp else self.world.get_map().get_waypoint(
                    vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            else:
                waypoint = self.world.get_map().get_waypoint(
                    vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)

            image = process_img(image_rgb)
            next_state = image

            while True:
                if self.visuals:
                    camera_switch = check_camera_switch()
                    if camera_switch == -1:
                        return False
                    elif camera_switch > 0:
                        self.current_camera_mode = camera_switch
                        print(f"Camera mode: {camera_switch}")
                    self.clock.tick_busy_loop(30)

                vehicle_location = self.vehicle.get_location()

                # Route-aware waypoint
                if self.route_waypoints:
                    route_wp = self._get_next_route_waypoint()
                    waypoint = route_wp if route_wp else self.world.get_map().get_waypoint(
                        vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
                else:
                    waypoint = self.world.get_map().get_waypoint(
                        vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)

                speed = get_speed(self.vehicle)

                state = next_state
                counter += 1
                self.global_t += 1

                # Select action
                action = model.select_action(state, eval=eval)
                steer = action
                if action_map is not None:
                    steer = action_map[action]

                # Radar proximity check (ignore very close = likely car's own hood)
                min_dist = 100.0
                if radar_data:
                    for detect in radar_data:
                        if 1.5 < detect.depth < min_dist:  # Ignore objects < 1.5m (car's own body)
                            min_dist = detect.depth

                # Traffic light detection
                traffic_light_state = None
                is_red_light = False
                try:
                    traffic_light_state = self.vehicle.get_traffic_light_state()
                    if self.vehicle.is_at_traffic_light():
                        if traffic_light_state == carla.TrafficLightState.Red:
                            is_red_light = True
                except:
                    pass

                # Emergency stop: only brake briefly, then resume
                emergency_stop = False
                if is_red_light and speed > 3.0:
                    emergency_stop = True
                elif min_dist < 3.0 and speed > 10.0:
                    emergency_stop = True

                # Apply control
                if emergency_stop:
                    control = carla.VehicleControl(throttle=0.0, steer=steer, brake=1.0)
                else:
                    control = self.speed_controller.run_step(self.target_speed)
                    control.steer = steer
                self.vehicle.apply_control(control)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Tick next frame
                tick_data = sync_mode.tick(timeout=0.5)
                if not tick_data or tick_data[0] is None:
                    print("Tick timeout")
                    break

                snapshot, image_rgb, image_rgb_vis, image_top, image_side, \
                    image_sem, image_depth, lidar_data, radar_data, gnss_data, imu_data, collision = tick_data

                collision_val = 0 if collision is None else 1

                # Pedestrian collision check
                is_pedestrian = False
                if collision is not None:
                    try:
                        if 'walker' in collision.other_actor.type_id:
                            is_pedestrian = True
                            print("PEDESTRIAN COLLISION!")
                    except:
                        pass

                cos_yaw_diff, dist, _ = get_reward_comp(self.vehicle, waypoint, collision)
                reward = reward_value(cos_yaw_diff, dist, collision_val)

                # Lane-keeping: ALWAYS follow the actual road centerline
                road_waypoint = self.world.get_map().get_waypoint(
                    vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
                if road_waypoint:
                    road_dist = vehicle_location.distance(road_waypoint.transform.location)
                    # Yaw alignment with road
                    road_cos = np.cos((correct_yaw(self.vehicle.get_transform().rotation.yaw) - 
                                       correct_yaw(road_waypoint.transform.rotation.yaw)) * np.pi / 180.)
                    
                    # Road following is the PRIMARY reward
                    reward = 10.0 * road_cos          # Strong: stay aligned with road
                    reward += 20.0 / (1.0 + road_dist)  # Stronger: stay centered on road
                    
                    # Penalty for going off-road (exponential)
                    if road_dist > 2.0:
                        reward -= 100.0 * (road_dist - 2.0)
                else:
                    reward = -200.0  # Way off road, no waypoint found

                # Proximity penalty
                proximity_penalty = -10.0 * (1.0 - min_dist / 10.0) if min_dist < 10.0 else 0.0
                pedestrian_penalty = -500.0 if is_pedestrian else 0.0
                light_penalty = -20.0 if is_red_light and speed > 2.0 else 0.0

                # Route progress bonus
                route_bonus = 0.0
                if self.route_waypoints:
                    progress = self._get_route_progress()
                    route_bonus = 10.0 * progress

                reward += proximity_penalty + pedestrian_penalty + light_penalty + route_bonus

                goal_reached = self._is_goal_reached()
                if goal_reached:
                    reward += 500.0
                    print(f"GOAL REACHED! Steps: {counter}")

                if image_rgb is None:
                    print("Image lost")
                    break

                image = process_img(image_rgb)
                done = 1 if collision_val else 0
                self.total_rewards += reward
                next_state = image

                replay_buffer.add(state, action, next_state, reward, done)

                if not eval:
                    if ep > self.start_train and (self.global_t % self.train_freq) == 0:
                        model.train(replay_buffer)

                # ── Pygame Display ──────────────────────────────────
                if self.visuals:
                    display_image = None
                    camera_name = ""

                    if self.current_camera_mode == 1:
                        display_image = image_rgb
                        camera_name = "Front Camera (AI)"
                    elif self.current_camera_mode == 2:
                        display_image = image_rgb_vis
                        camera_name = "Third-Person"
                    elif self.current_camera_mode == 3:
                        display_image = image_top
                        camera_name = "Top-Down"
                    elif self.current_camera_mode == 4:
                        display_image = image_side
                        camera_name = "Side View"
                    elif self.current_camera_mode == 5:
                        image_sem.convert(carla.ColorConverter.CityScapesPalette)
                        display_image = image_sem
                        camera_name = "Semantic Seg"
                    elif self.current_camera_mode == 6:
                        image_depth.convert(carla.ColorConverter.LogarithmicDepth)
                        display_image = image_depth
                        camera_name = "Depth View"
                    elif self.current_camera_mode == 7:
                        # Lidar visualization
                        lidar_surface = pygame.Surface((800, 600))
                        lidar_surface.fill((0, 0, 0))
                        if lidar_data:
                            points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
                            points = np.reshape(points, (int(points.shape[0] / 4), 4))
                            for radius in [10, 20, 30]:
                                pygame.draw.circle(lidar_surface, (40, 40, 40), (400, 300), radius * 10, 1)
                            pygame.draw.line(lidar_surface, (30, 30, 30), (0, 300), (800, 300), 1)
                            pygame.draw.line(lidar_surface, (30, 30, 30), (400, 0), (400, 600), 1)
                            for p in points:
                                x, y = int(400 + p[1] * 10), int(300 - p[0] * 10)
                                if 0 <= x < 800 and 0 <= y < 600:
                                    if p[2] > 1.0:
                                        color = (255, 255, 255)
                                    elif p[2] > -0.5:
                                        color = (0, 255, 0)
                                    else:
                                        color = (0, 100, 0)
                                    lidar_surface.set_at((x, y), color)
                        pygame.draw.polygon(lidar_surface, (0, 100, 255), [(395, 305), (405, 305), (400, 290)])
                        self.display.blit(lidar_surface, (0, 0))
                        camera_name = "Lidar Scan"
                    elif self.current_camera_mode == 8:
                        # Radar view - show third person with radar overlay
                        display_image = image_rgb_vis
                        camera_name = "Radar View"
                    elif self.current_camera_mode == 9:
                        # GPS / IMU view - third person background with data overlay
                        display_image = image_rgb_vis
                        camera_name = "GPS / IMU"

                    if self.current_camera_mode != 7 and display_image is not None:
                        draw_image(self.display, display_image)

                    # Camera name
                    self.display.blit(
                        self.font.render(f'Camera: {camera_name}', True, (255, 255, 0)),
                        (8, 10))
                    self.display.blit(
                        self.font.render('Keys: 1-4=Cam 5=Sem 6=Depth 7=Lidar 8=Radar 9=GPS/IMU', True, (200, 200, 200)),
                        (8, 25))
                    self.display.blit(
                        self.font.render('FPS: %.1f (sim %d)' % (self.clock.get_fps(), fps), True, (255, 255, 255)),
                        (8, 40))

                    # Route progress
                    if self.route_waypoints:
                        progress = self._get_route_progress() * 100
                        self.display.blit(
                            self.font.render('Route: %.1f%% (%d/%d wp)' % (
                                progress, self.current_route_index, len(self.route_waypoints)), True, (0, 255, 255)),
                            (8, 55))
                        self.display.blit(
                            self.font.render('Route dist: %.1f m' % dist, True, (0, 255, 255)),
                            (8, 70))

                    # Obstacle distance - position below route info
                    obstacle_y = 85 if self.route_waypoints else 70
                    self.display.blit(
                        self.font.render('Obstacle: %.1f m' % min_dist, True,
                            (255, 0, 0) if min_dist < 10 else (0, 255, 0)),
                        (8, obstacle_y))

                    # ── Dashboard panel ──────────────────────────────
                    panel_x, panel_y, panel_w, panel_h = 600, 10, 190, 180
                    pygame.draw.rect(self.display, (0, 0, 0), (panel_x, panel_y, panel_w, panel_h))
                    pygame.draw.rect(self.display, (255, 255, 255), (panel_x, panel_y, panel_w, panel_h), 1)

                    self.display.blit(self.font.render('AI DASHBOARD', True, (0, 255, 255)), (panel_x + 10, panel_y + 5))

                    # Steering
                    steer_color = (0, 255, 0) if abs(steer) < 0.1 else (255, 255, 0)
                    if abs(steer) > 0.5:
                        steer_color = (255, 0, 0)
                    self.display.blit(self.font.render('Steer: %.2f' % steer, True, steer_color), (panel_x + 10, panel_y + 25))

                    # Status
                    status_text = "CRUISING"
                    status_color = (0, 255, 0)
                    if emergency_stop:
                        status_text = "RED LIGHT" if is_red_light else "OBSTACLE"
                        status_color = (255, 0, 0)
                    elif speed < 5:
                        status_text = "STARTING"
                    self.display.blit(self.font.render('Status: %s' % status_text, True, status_color), (panel_x + 10, panel_y + 45))

                    # Speed
                    self.display.blit(self.font.render('Speed: %.1f km/h' % speed, True, (255, 255, 255)), (panel_x + 10, panel_y + 65))

                    # Traffic light
                    tl_text = str(traffic_light_state).split('.')[-1] if traffic_light_state else "None"
                    tl_color = (0, 255, 0)
                    if traffic_light_state == carla.TrafficLightState.Red:
                        tl_color = (255, 0, 0)
                    elif traffic_light_state == carla.TrafficLightState.Yellow:
                        tl_color = (255, 255, 0)
                    self.display.blit(self.font.render('Light: %s' % tl_text, True, tl_color), (panel_x + 10, panel_y + 85))

                    # Lane alignment
                    alignment = (cos_yaw_diff + 1) / 2 * 100
                    self.display.blit(self.font.render('Lane Align: %.1f%%' % alignment, True, (255, 255, 255)), (panel_x + 10, panel_y + 105))
                    self.display.blit(self.font.render('Lane Dist: %.2f m' % dist, True, (255, 255, 255)), (panel_x + 10, panel_y + 125))

                    # Step counter
                    self.display.blit(self.font.render('Step: %d' % counter, True, (200, 200, 200)), (panel_x + 10, panel_y + 145))

                    # Reward
                    self.display.blit(self.font.render('Reward: %.1f' % reward, True, (200, 200, 200)), (panel_x + 10, panel_y + 163))

                    # GPS / IMU overlay when mode 9
                    if self.current_camera_mode == 9:
                        imu_panel_x, imu_panel_y = 10, 100
                        imu_panel_w, imu_panel_h = 260, 150
                        pygame.draw.rect(self.display, (0, 0, 0), (imu_panel_x, imu_panel_y, imu_panel_w, imu_panel_h))
                        pygame.draw.rect(self.display, (255, 255, 0), (imu_panel_x, imu_panel_y, imu_panel_w, imu_panel_h), 1)

                        self.display.blit(self.font.render('LOCATION & INERTIA', True, (255, 255, 0)), (imu_panel_x + 10, imu_panel_y + 5))

                        lat = gnss_data.latitude if gnss_data else 0.0
                        lon = gnss_data.longitude if gnss_data else 0.0
                        self.display.blit(self.font.render('Lat: %.6f' % lat, True, (255, 255, 255)), (imu_panel_x + 10, imu_panel_y + 25))
                        self.display.blit(self.font.render('Lon: %.6f' % lon, True, (255, 255, 255)), (imu_panel_x + 10, imu_panel_y + 45))

                        if imu_data:
                            accel_x = imu_data.accelerometer.x
                            accel_y = imu_data.accelerometer.y
                            accel_z = imu_data.accelerometer.z
                            gyro_x = imu_data.gyroscope.x
                            gyro_y = imu_data.gyroscope.y
                            gyro_z = imu_data.gyroscope.z
                        else:
                            accel_x = accel_y = accel_z = gyro_x = gyro_y = gyro_z = 0.0

                        self.display.blit(self.font.render('Accel X: %.2f m/s2' % accel_x, True, (0, 255, 255)), (imu_panel_x + 10, imu_panel_y + 70))
                        self.display.blit(self.font.render('Accel Y: %.2f m/s2' % accel_y, True, (0, 255, 255)), (imu_panel_x + 10, imu_panel_y + 88))
                        self.display.blit(self.font.render('Accel Z: %.2f m/s2' % accel_z, True, (0, 255, 255)), (imu_panel_x + 10, imu_panel_y + 106))
                        self.display.blit(self.font.render('Gyro  Z: %.2f rad/s' % gyro_z, True, (255, 150, 255)), (imu_panel_x + 10, imu_panel_y + 126))

                    pygame.display.flip()

                # Termination conditions
                if collision_val == 1:
                    print("Episode {} - COLLISION at step {}".format(ep, counter))
                    break
                if goal_reached:
                    print("Episode {} - GOAL REACHED at step {}".format(ep, counter))
                    break
                if counter >= self.max_iter:
                    print("Episode {} - max iter reached".format(ep))
                    break
                # Only check off-road when NOT following a route
                if not self.route_waypoints and dist > 50.0:
                    print("Episode {} - very off road ({}m)".format(ep, dist))
                    break

            if ep % self.save_freq == 0 and ep > 0:
                self.save(model, ep)

    def save(self, model, ep):
        if ep % self.save_freq == 0 and ep > self.start_ep:
            avg_reward = self.total_rewards / self.save_freq
            self.average_rewards_list.append(avg_reward)
            self.total_rewards = 0
            model.save('weights/model_ep_{}'.format(ep))
            print("Saved model with average reward =", avg_reward)


def get_reward_comp(vehicle, waypoint, collision):
    vehicle_location = vehicle.get_location()
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y
    x_vh = vehicle_location.x
    y_vh = vehicle_location.y
    wp_array = np.array([x_wp, y_wp])
    vh_array = np.array([x_vh, y_vh])
    dist = np.linalg.norm(wp_array - vh_array)
    vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
    wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
    cos_yaw_diff = np.cos((vh_yaw - wp_yaw) * np.pi / 180.)
    collision = 0 if collision is None else 1
    return cos_yaw_diff, dist, collision


def reward_value(cos_yaw_diff, dist, collision, lambda_1=1, lambda_2=1, lambda_3=5):
    reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision)
    return reward
