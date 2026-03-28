import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def force_cleanup():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        print("Synchronous mode disabled.")

        world.tick()

        print("Cleaning up actors...")
        for actor in world.get_actors():
            if any(x in actor.type_id for x in ['vehicle', 'sensor', 'walker', 'controller']):
                try:
                    if actor.is_alive:
                        actor.destroy()
                except:
                    pass

        print("Reloading Town02_Opt...")
        client.load_world('Town02_Opt')
        print("Force cleanup completed!")
    except Exception as e:
        print(f"Force cleanup failed: {e}")


if __name__ == "__main__":
    force_cleanup()
