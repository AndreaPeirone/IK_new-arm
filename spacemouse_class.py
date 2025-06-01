import pywinusb.hid as hid
import time
import threading

class SpaceMouse:
    def __init__(self, vendor_id=0x256F, product_id=0xC652, device_index=3):
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device_index = device_index
        self.device = None
        self._data = [0] * 8  # [x, y, z, roll, pitch, yaw, button1, button2]
        self._lock = threading.Lock()
        self._running = False

    def _handler(self, data):
        report_id = data[0]
        with self._lock:
            if report_id == 1 and len(data) >= 13:
                self._data[0] = int.from_bytes(data[1:3], 'little', signed=True)
                self._data[1] = int.from_bytes(data[3:5], 'little', signed=True)
                self._data[2] = int.from_bytes(data[5:7], 'little', signed=True)
                self._data[3] = int.from_bytes(data[7:9], 'little', signed=True)
                self._data[4] = int.from_bytes(data[9:11], 'little', signed=True)
                self._data[5] = int.from_bytes(data[11:13], 'little', signed=True)
            elif report_id == 3 and len(data) >= 2:
                buttons = data[1]
                self._data[6] = buttons & 1
                self._data[7] = (buttons >> 1) & 1

    def start(self):
        devices = hid.HidDeviceFilter(vendor_id=self.vendor_id, product_id=self.product_id).get_devices()
        if len(devices) <= self.device_index:
            raise IndexError(f"Device index {self.device_index} not found.")
        self.device = devices[self.device_index]
        self.device.open()
        self.device.set_raw_data_handler(self._handler)
        self._running = True

    def stop(self):
        self._running = False
        if self.device:
            self.device.close()
            self.device = None

    def get_array(self):
        with self._lock:
            x, y, z, roll, pitch, yaw, b1, b2 = self._data
            return [x*1e-3, -y*1e-3, -z*1e-3, roll*1e-3, -pitch*1e-3, -yaw*1e-3, b1, b2]

    def is_running(self):
        return self._running



# import pywinusb.hid as hid
# import threading
# import time

# class SpaceMouse:
#     def __init__(self, vendor_id=0x256F, product_id=0xC652, device_index=None, test_timeout=2):
#         """
#         Initializes a SpaceMouse. If device_index is None, automatically detects the first working interface.
#         """
#         self.vendor_id = vendor_id
#         self.product_id = product_id
#         self.test_timeout = test_timeout
#         self.device_index = device_index
#         self.device = None
#         self._data = [0] * 8  # [x, y, z, roll, pitch, yaw, button1, button2]
#         self._lock = threading.Lock()
#         self._running = False

#         if self.device_index is None:
#             self.device_index = self._detect_index()
#             if self.device_index is None:
#                 raise IOError("Could not auto-detect a working SpaceMouse interface.")

#     def _handler(self, data):
#         report_id = data[0]
#         with self._lock:
#             if report_id == 1 and len(data) >= 13:
#                 # parse six-axis translation/rotation
#                 vals = [int.from_bytes(data[i:i+2], 'little', signed=True) for i in (1,3,5,7,9,11)]
#                 self._data[0:6] = vals
#             elif report_id == 3 and len(data) >= 2:
#                 buttons = data[1]
#                 self._data[6] = buttons & 1
#                 self._data[7] = (buttons >> 1) & 1

#     def _test_device(self, device):
#         data_received = False
#         def probe(data):
#             nonlocal data_received
#             if data[0] in (1,3):
#                 data_received = True
#         try:
#             device.open()
#             device.set_raw_data_handler(probe)
#             time.sleep(self.test_timeout)
#             return data_received
#         finally:
#             device.close()

#     def _detect_index(self):
#         devices = hid.HidDeviceFilter(vendor_id=self.vendor_id, product_id=self.product_id).get_devices()
#         for idx, dev in enumerate(devices):
#             if self._test_device(dev):
#                 return idx
#         return None

#     def start(self):
#         devices = hid.HidDeviceFilter(vendor_id=self.vendor_id, product_id=self.product_id).get_devices()
#         if self.device_index >= len(devices):
#             raise IndexError(f"Device index {self.device_index} not found.")
#         self.device = devices[self.device_index]
#         self.device.open()
#         self.device.set_raw_data_handler(self._handler)
#         self._running = True

#     def stop(self):
#         self._running = False
#         if self.device:
#             self.device.close()
#             self.device = None

#     def get_array(self):
#         """
#         Returns [x, y, z, roll, pitch, yaw, b1, b2] with units in meters/radians as floats.
#         """
#         with self._lock:
#             x, y, z, roll, pitch, yaw, b1, b2 = self._data
#         # scale and invert as needed
#         return [x*1e-3, -y*1e-3, -z*1e-3, roll*1e-3, -pitch*1e-3, -yaw*1e-3, b1, b2]

#     def is_running(self):
#         return self._running

