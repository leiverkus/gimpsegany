# Bringing GIMP 3 Python plug-in dialogs to the front on macOS

A copy-paste recipe for any GIMP 3 Python plug-in that opens a GTK dialog
and suffers from the two classic macOS problems:

1. The plug-in shows up as its own app in the Dock with a python-launcher
   rocket icon.
2. The dialog opens behind GIMP's main window and doesn't take focus, so
   the user has to click it before typing.

## Why the obvious fixes don't work

GIMP 3 runs each Python plug-in as a detached subprocess. On macOS this
subprocess has no `.app` bundle, so the system treats it as a generic
Python interpreter launch: rocket icon in the Dock, "Python" in the menu
bar, no window-manager relationship with the GIMP process.

GTK 3 on macOS uses the quartz backend, and almost every WM hint you
reach for is an X11-only no-op there:

| Call | Effect on quartz |
|---|---|
| `Gtk.Window.set_keep_above(True)` | none |
| `Gtk.Window.set_skip_taskbar_hint(True)` | none / inconsistent |
| `Gtk.Window.present_with_time()` | doesn't activate the app |
| `Gtk.Window.present()` | doesn't activate the app |
| `GLib.set_prgname()` | sets GLib's internal name only; never reaches `NSProcessInfo`, so the menu bar keeps saying "Python" |

The "proper" fix is
`NSApplication.sharedApplication().activateIgnoringOtherApps_(True)` via
PyObjC. But the Python interpreter bundled with GIMP 3 on macOS does
**not** ship PyObjC, and installing packages into a `GIMP.app` bundle is
fragile and user-hostile — the plug-in should work out of the box.

## The working fix: Carbon Process Manager via `ctypes`

The deprecated-but-still-present Carbon Process Manager APIs
(`GetCurrentProcess`, `TransformProcessType`, `SetFrontProcess`) are
exposed by the `ApplicationServices` umbrella framework and reachable
through the stdlib `ctypes` module. No extra dependencies, ~30 lines.

`TransformProcessType(psn, 4)` — where `4` is the Carbon constant
`kProcessTransformToUIElementApplication` — turns the current process
into an **accessory app**: no Dock icon, no app menu, but windows can
still receive focus and events. `SetFrontProcess` then raises it above
GIMP.

```python
import sys


def macos_make_accessory_and_front():
    """Hide the python-launcher Dock icon and pull the window to the front.

    Call once after the plug-in's main window is mapped (e.g. from a
    ``map-event`` handler). Safe no-op on non-macOS platforms and on
    macOS versions where the Carbon symbols have been removed.
    """
    if sys.platform != "darwin":
        return
    try:
        import ctypes

        fw = ctypes.CDLL(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )

        class ProcessSerialNumber(ctypes.Structure):
            _fields_ = [
                ("highLongOfPSN", ctypes.c_uint32),
                ("lowLongOfPSN", ctypes.c_uint32),
            ]

        fw.GetCurrentProcess.argtypes = [ctypes.POINTER(ProcessSerialNumber)]
        fw.GetCurrentProcess.restype = ctypes.c_int32
        fw.TransformProcessType.argtypes = [
            ctypes.POINTER(ProcessSerialNumber),
            ctypes.c_uint32,
        ]
        fw.TransformProcessType.restype = ctypes.c_int32
        fw.SetFrontProcess.argtypes = [ctypes.POINTER(ProcessSerialNumber)]
        fw.SetFrontProcess.restype = ctypes.c_int32

        psn = ProcessSerialNumber(0, 0)
        if fw.GetCurrentProcess(ctypes.byref(psn)) != 0:
            return
        fw.TransformProcessType(ctypes.byref(psn), 4)  # UIElement / accessory
        fw.SetFrontProcess(ctypes.byref(psn))
    except Exception:
        pass
```

## Where to call it

Call it from a `map-event` handler on the dialog, **not** from
`__init__`. At `__init__` time the window isn't realised yet and Carbon
can't raise a process that has no on-screen windows.

```python
import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk


class MyDialog(Gtk.Dialog):
    def __init__(self):
        Gtk.Dialog.__init__(
            self,
            title="My Plugin",
            transient_for=None,
            flags=Gtk.DialogFlags.MODAL,
        )
        self.set_modal(True)
        # ...build the dialog...
        self.connect("map-event", self._on_map_event)
        self.show_all()

    def _on_map_event(self, widget, event):
        macos_make_accessory_and_front()
```

`flags=Gtk.DialogFlags.MODAL` (or `self.set_modal(True)`) is worth
setting alongside — it's cheap, gets you proper modality on Linux where
Carbon isn't involved, and pairs nicely with the accessory-mode fix on
macOS.

## Caveats

- **Deprecated APIs.** `GetCurrentProcess` / `TransformProcessType` /
  `SetFrontProcess` have been deprecated since macOS 10.9. They still
  resolve and work on macOS 14 Sonoma and 15 Sequoia (tested), but may
  be removed in a future major release. When that happens the
  `try/except` silently no-ops and you're back to the old "rocket icon,
  dialog behind GIMP" state — nothing worse.
- **Focus works, Dock-icon removal is less reliable.** The front-raise
  is very reliable. The accessory-mode transform (Dock icon removal)
  works on some GIMP 3 builds and not others, depending on how the
  python subprocess was spawned. Test on your target GIMP build.
- **macOS only.** Linux uses X11/Wayland WM hints via GTK directly and
  doesn't need any of this. Windows also has a different story.
- **Menu bar name.** Accessory apps don't have an app menu at all, so
  "Python" disappears rather than being replaced with your plug-in
  name. `GLib.set_prgname()` and `GLib.set_application_name()` are
  still worth calling for Linux's sake and don't hurt.
- **No PyObjC path.** If some future GIMP build happens to ship PyObjC,
  `NSApplication.sharedApplication().setActivationPolicy_(1)` +
  `activateIgnoringOtherApps_(True)` is the more-correct equivalent.
  You can add a PyObjC branch before the ctypes one as a preference,
  keeping the ctypes path as the fallback.

## Why this isn't a shared library

Because it's one function, zero dependencies, and the copy-paste
overhead is smaller than the import overhead. Just drop the function
into your plug-in and wire up the `map-event` handler.
