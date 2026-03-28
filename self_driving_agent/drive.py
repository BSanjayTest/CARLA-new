import os
import sys
import math
import argparse
import numpy as np

# Find CARLA agents path
_carla_root = os.environ.get('CARLA_ROOT', '')
_carla_agents = os.path.join(_carla_root, 'PythonAPI', 'carla') if _carla_root else ''
if not _carla_agents or not os.path.exists(_carla_agents):
    for _drive in ['C', 'D', 'E']:
        for _folder in ['CARLA', 'CARLA_0.9.16', 'selfdriving', 'carla']:
            _test = f'{_drive}:\\{_folder}\\PythonAPI\\carla'
            if os.path.exists(_test):
                _carla_agents = _test
                break
        if _carla_agents:
            break
if _carla_agents:
    sys.path.append(_carla_agents)

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

parser = argparse.ArgumentParser(description='PID route-following drive')
parser.add_argument('--start', type=int, default=0, help='Spawn point index')
parser.add_argument('--goal', type=int, default=0, help='Goal index (same=start=round trip)')
args = parser.parse_args()

# PID Controllers
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def step(self, error, dt=0.05):
        self.integral += error * dt
        self.integral = max(-5, min(5, self.integral))  # anti-windup
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def get_route(world, start_idx, goal_idx):
    sp = world.get_map().get_spawn_points()
    grp = GlobalRoutePlanner(world.get_map(), 2.0)

    if start_idx == goal_idx:
        # Round trip: go to farthest point and back
        max_dist = 0
        far_idx = 0
        for i, p in enumerate(sp):
            if i == start_idx:
                continue
            d = sp[start_idx].location.distance(p.location)
            if d > max_dist:
                max_dist = d
                far_idx = i
        leg1 = grp.trace_route(sp[start_idx].location, sp[far_idx].location)
        leg2 = grp.trace_route(sp[far_idx].location, sp[start_idx].location)
        route = list(leg1) + list(leg2)
        print(f"Round trip: {len(route)} waypoints, turnaround at point {far_idx}")
    else:
        route = grp.trace_route(sp[start_idx].location, sp[goal_idx].location)
        print(f"Route: {len(route)} waypoints")
    return route


def draw_hud(display, font, clock, speed, steer, wp_idx, total_wps, dist, tl_state):
    import pygame
    display.fill((0, 0, 0))

    lines = [
        f'PID Route Follower',
        f'Speed: {speed:.1f} km/h',
        f'Steer: {steer:.2f}',
        f'Waypoint: {wp_idx}/{total_wps}',
        f'Dist to WP: {dist:.1f} m',
        f'Light: {tl_state}',
        f'FPS: {clock.get_fps():.1f}',
    ]
    for i, line in enumerate(lines):
        color = (0, 255, 0)
        if 'RED' in tl_state:
            color = (255, 0, 0)
        elif 'YELLOW' in tl_state:
            color = (255, 255, 0)
        display.blit(font.render(line, True, color), (10, 10 + i * 22))
    pygame.display.flip()


def main():
    import pygame

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town02_Opt')

    # Fast traffic lights
    for tl in world.get_actors().filter('traffic.traffic_light'):
        tl.set_red_time(2.0)
        tl.set_green_time(4.0)
        tl.set_yellow_time(1.0)

    spawn_points = world.get_map().get_spawn_points()

    # Generate route
    route = get_route(world, args.start, args.goal)
    waypoints = [wp for (wp, _) in route]

    # Spawn vehicle
    blueprint = world.get_blueprint_library().find('vehicle.nissan.patrol')
    vehicle = world.spawn_actor(blueprint, spawn_points[args.start])
    print(f"Spawned at point {args.start}")

    # PIDs
    steer_pid = PIDController(kp=1.5, ki=0.0, kd=0.3)
    speed_pid = PIDController(kp=0.5, ki=0.01, kd=0.1)
    target_speed = 25.0  # km/h

    # Pygame
    pygame.init()
    display = pygame.display.set_mode((400, 250))
    font = pygame.font.SysFont('monospace', 18)
    clock = pygame.time.Clock()

    wp_index = 0
    try:
        while wp_index < len(waypoints):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                    raise KeyboardInterrupt

            loc = vehicle.get_location()
            yaw = vehicle.get_transform().rotation.yaw

            # Current speed
            vel = vehicle.get_velocity()
            speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            # Find next waypoint ahead
            while wp_index < len(waypoints) - 1:
                wp_loc = waypoints[wp_index].transform.location
                if loc.distance(wp_loc) > 3.0:
                    break
                wp_index += 1

            if wp_index >= len(waypoints):
                break

            wp = waypoints[wp_index]
            wp_loc = wp.transform.location
            wp_yaw = wp.transform.rotation.yaw

            # Distance to waypoint
            dist = loc.distance(wp_loc)

            # Steering: PID on yaw error
            dx = wp_loc.x - loc.x
            dy = wp_loc.y - loc.y
            target_yaw = math.degrees(math.atan2(dy, dx))
            yaw_error = target_yaw - yaw
            while yaw_error > 180: yaw_error -= 360
            while yaw_error < -180: yaw_error += 360

            steer = max(-1.0, min(1.0, steer_pid.step(yaw_error / 90.0)))

            # Speed control
            speed_error = target_speed - speed
            throttle = max(0.0, min(1.0, speed_pid.step(speed_error)))

            # Traffic light check
            tl_state = "Green"
            try:
                is_at_tl = vehicle.is_at_traffic_light()
                if is_at_tl:
                    tl = vehicle.get_traffic_light_state()
                    tl_state = str(tl).split('.')[-1]
                    if tl == carla.TrafficLightState.Red:
                        throttle = 0.0
                        brake = 1.0
                    elif tl == carla.TrafficLightState.Yellow:
                        throttle = 0.0
                        brake = 0.5
                    else:
                        brake = 0.0
                else:
                    brake = 0.0
            except:
                brake = 0.0

            control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            vehicle.apply_control(control)

            draw_hud(display, font, clock, speed, steer, wp_index, len(waypoints), dist, tl_state)
            clock.tick(30)
            world.tick()

        print(f"ROUTE COMPLETED! {len(waypoints)} waypoints driven.")
        print("Press ESC or close window to exit.")

        # Keep window open
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                    raise KeyboardInterrupt
            clock.tick(30)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            vehicle.destroy()
        except:
            pass
        pygame.quit()
        print("Done.")


if __name__ == "__main__":
    main()
