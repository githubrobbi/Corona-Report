import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import os
import git
import datetime
import Corona

watchDirectory = r"D:/DATA/Protect/IMPORTANT/Python/Harvard Course/HomeWork/Data/Corona by Johns Hopkins/csse_covid_19_data/csse_covid_19_time_series"
Corona_Dir = r"D:/DATA/Protect/IMPORTANT/Python/Harvard Course/HomeWork/Data/Corona by Johns Hopkins"
Triggered = False

def on_created(event):
    date_stamp = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
    print("Watchdog received CREATE event - % s." % event.src_path)
    print(date_stamp)
    global Triggered
    if not Triggered:
        Corona.Corona_App()
        print("created")
        Triggered = True


def on_deleted(event):
    date_stamp = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
    print("Watchdog received DELETE event - % s." % event.src_path)
    print(date_stamp)
    global Triggered
    if not Triggered:
        Corona.Corona_App()
        print("deleted")
        Triggered = True

def on_modified(event):
    date_stamp = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
    print("Watchdog received MODIFY event - % s." % event.src_path)
    print(date_stamp)
    global Triggered
    if not Triggered:
        Corona.Corona_App()
        print("modified")
        Triggered = True

def on_moved(event):
    date_stamp = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
    print("Watchdog received MOVE event - % s." % event.src_path)
    print(date_stamp)
    global Triggered
    if not Triggered:
        Corona.Corona_App()
        print("moved")
        Triggered = True

def on_any(event):
    date_stamp = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
    print("Watchdog received ANY event - % s." % event.src_path)
    print(date_stamp)
    global Triggered
    if not Triggered:
        Corona.Corona_App()
        print("any")
        Triggered = True


if __name__ == "__main__":
    patterns = ['*.csv']
    ignore_patterns = None #['*.md']
    ignore_directories = True
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)

    my_event_handler.on_created = on_created
    my_event_handler.on_deleted = on_deleted
    my_event_handler.on_modified = on_modified
    my_event_handler.on_moved = on_moved
    # my_event_handler.on_any_event = on_any

    path = watchDirectory
    go_recursively = True
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)

    my_observer.start()
    try:
        while True:
            Triggered = False
            if os.path.exists(Corona_Dir):

                result = git.Repo(Corona_Dir).remotes.origin.pull()

                if git.Repo(Corona_Dir).is_dirty(untracked_files=True):
                    # git.Repo.clone_from('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master', depth=1)
                    # git.Remote.update('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master', depth=1)
                    # git.Submodule.update(recursive=False, init=True, to_latest_revision=True, progress=None, dry_run=False, force=False,
                    #        keep_going=False)
                    os.system('rmdir "../Data/Corona by Johns Hopkins" /s/q')

                    # Create temporary dir
                    os.mkdir(Corona_Dir)

                    # Clone into temporary dir
                    git.Repo.clone_from('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master',
                                        depth=1)
                    # Corona.Corona_App()
                    print("Dirty App")

            else:  # NO Dir

                # Create temporary dir
                os.mkdir(Corona_Dir)

                # Clone into temporary dir
                git.Repo.clone_from('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master',
                                    depth=1)
                Corona.Corona_App()
                print("No Dir App")

            time.sleep(600)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()


# print("Landed")
#
# Corona_Dir = "../Data/Corona by Johns Hopkins"
# # watchDirectory = "../Data/Corona by Johns Hopkins/archived_data/archived_time_series"
#
# watchFile = "time_series_2019-ncov-Deaths.csv"
# Triggered = False
# workingdir = r"D:\DATA\Protect\IMPORTANT\Python\Harvard Course\HomeWork\Source"
#
# os.chdir(workingdir)
# import pathlib
# print(pathlib.Path().absolute())
#
# class Handler(watchdog.events.PatternMatchingEventHandler):
#     def __init__(self):
#         # Set the patterns for PatternMatchingEventHandler
#         watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['*.csv'],
#                                                              ignore_directories=True, case_sensitive=False)
#
#     def on_created(self, event):
#         date_stamp = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
#         print("Watchdog received modified event - % s." % event.src_path)
#         print(date_stamp)
#         global Triggered
#         # Event is created, you can process it now
#         if not Triggered:
#             # os.system('python Corona.py')
#             Corona.Corona_App()
#             print("created")
#             Triggered = True
#
#     def on_modified(self, event):
#         date_stamp = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
#         print("Watchdog received modified event - % s." % event.src_path)
#         print(date_stamp)
#         global Triggered
#         # Event is modified, you can process it now
#         if not Triggered:
#             # os.system('python Corona.py')
#             Corona.Corona_App()
#             print("modified")
#             Triggered = True
#
# if __name__ == "__main__":
#     if os.path.exists(Corona_Dir):
#
#         result = git.Repo(Corona_Dir).remotes.origin.pull()
#
#         if git.Repo(Corona_Dir).is_dirty(untracked_files=True):
#             # git.Repo.clone_from('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master', depth=1)
#             # git.Remote.update('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master', depth=1)
#             # git.Submodule.update(recursive=False, init=True, to_latest_revision=True, progress=None, dry_run=False, force=False,
#             #        keep_going=False)
#             os.system('rmdir "../Data/Corona by Johns Hopkins" /s/q')
#
#             # Create temporary dir
#             os.mkdir(Corona_Dir)
#
#             # Clone into temporary dir
#             git.Repo.clone_from('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master',
#                                 depth=1)
#
#             Corona.Corona_App()
#
#
#     else:  # NO Dir
#
#         # Create temporary dir
#         os.mkdir(Corona_Dir)
#
#         # Clone into temporary dir
#         git.Repo.clone_from('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master',
#                             depth=1)
#
#         Corona.Corona_App()
#
#     src_path = watchDirectory
#     event_handler = Handler()
#     observer = watchdog.observers.Observer()
#     observer.schedule(event_handler, path=watchDirectory, recursive=True)
#     observer.start()
#     try:
#         counter = 0
#         while True:
#             Triggered = False
#             if os.path.exists(Corona_Dir):
#
#                 result = git.Repo(Corona_Dir).remotes.origin.pull()
#
#                 if git.Repo(Corona_Dir).is_dirty(untracked_files=True):
#                     # git.Repo.clone_from('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master', depth=1)
#                     # git.Remote.update('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master', depth=1)
#                     # git.Submodule.update(recursive=False, init=True, to_latest_revision=True, progress=None, dry_run=False, force=False,
#                     #        keep_going=False)
#                     os.system('rmdir "../Data/Corona by Johns Hopkins" /s/q')
#
#                     # Create temporary dir
#                     os.mkdir(Corona_Dir)
#
#                     # Clone into temporary dir
#                     git.Repo.clone_from('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master',
#                                         depth=1)
#                     Corona.Corona_App()
#
#
#             else:  # NO Dir
#
#                 # Create temporary dir
#                 os.mkdir(Corona_Dir)
#
#                 # Clone into temporary dir
#                 git.Repo.clone_from('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master',
#                                     depth=1)
#                 Corona.Corona_App()
#
#             time.sleep(10)
#
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()
#
