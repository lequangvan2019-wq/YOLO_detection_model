#!/usr/bin/env python3
"""
Nokia 5G 360 Camera OSC API Client.

Provides direct camera control via the Open Spherical Camera (OSC) API.
Used for camera configuration, status monitoring, and control.

Based on Nokia 5G 360 Camera API Documentation v1.0 (September 2025)

Authentication: Digest Access Authentication (SHA-256)
Default credentials: Nokia360:Nokia360
"""

import os
import requests
from requests.auth import HTTPDigestAuth
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import urllib3

# Suppress InsecureRequestWarning for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


@dataclass
class BatteryStatus:
    """Camera battery status."""
    level: float  # 0.0 to 1.0, -1 if not present
    temperature_c: int
    is_present: bool

    @property
    def percentage(self) -> int:
        """Battery level as percentage (0-100)."""
        if self.level < 0:
            return -1
        return int(self.level * 100)


@dataclass
class CameraState:
    """Camera state from /osc/state endpoint."""
    fingerprint: str
    battery_level: float
    storage_uri: str
    gps_status: bool
    wifi_status: bool
    ethernet_status: bool
    cellular_status: bool
    temperature_5g_c: int
    temperature_battery_c: int
    temperature_sensor_min_c: int
    temperature_sensor_max_c: int
    oos_status: str  # Out of service status


class CameraOSCClient:
    """
    Client for Nokia 5G 360 Camera OSC API.

    Provides camera control including:
    - Status monitoring (battery, temperature, connectivity)
    - Live streaming control (RTSP/SRT)
    - Video/audio configuration
    - Capture control (image/video)

    Example:
        >>> camera = CameraOSCClient("192.168.18.19")
        >>> status = camera.get_state()
        >>> print(f"Battery: {camera.get_battery_status().percentage}%")
        >>> camera.set_video_settings(bitrate=20, resolution="8k", framerate=30)
    """

    # Default credentials per Nokia documentation
    DEFAULT_USERNAME = "Nokia360"
    DEFAULT_PASSWORD = "Nokia360"
    DEFAULT_PORT = 443

    def __init__(
        self,
        host: str,
        port: int = None,
        username: str = None,
        password: str = None,
        timeout: int = 10,
        use_https: bool = True,
        verify_ssl: bool = False
    ):
        """
        Initialize camera OSC client.

        Args:
            host: Camera IP address.
            port: Camera API port (default 443 for HTTPS).
            username: API username (default: Nokia360).
            password: API password (default: Nokia360).
            timeout: Request timeout in seconds.
            use_https: Use HTTPS (required for Nokia camera).
            verify_ssl: Verify SSL certificate (False for self-signed).
        """
        self.host = host
        self.port = port or self.DEFAULT_PORT
        self.username = username or os.getenv("CAMERA_OSC_USERNAME", self.DEFAULT_USERNAME)
        self.password = password or os.getenv("CAMERA_OSC_PASSWORD", self.DEFAULT_PASSWORD)
        self.timeout = timeout
        self.use_https = use_https
        self.verify_ssl = verify_ssl

        # Build base URL
        protocol = "https" if use_https else "http"
        self.base_url = f"{protocol}://{host}:{self.port}"

        # Setup session with digest auth
        self.session = requests.Session()
        self.auth = HTTPDigestAuth(self.username, self.password)

        logger.info(f"Camera OSC client initialized: {self.base_url}")

    def _get(self, endpoint: str) -> Dict:
        """
        Make authenticated GET request.

        Args:
            endpoint: API endpoint (e.g., "/osc/info").

        Returns:
            Response JSON as dict.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(
                url,
                auth=self.auth,
                timeout=self.timeout,
                verify=self.verify_ssl,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"GET {endpoint} failed: {e}")
            return {"error": str(e)}

    def _post(self, endpoint: str, payload: Dict = None) -> Dict:
        """
        Make authenticated POST request.

        Args:
            endpoint: API endpoint.
            payload: JSON payload.

        Returns:
            Response JSON as dict.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(
                url,
                auth=self.auth,
                json=payload,
                timeout=self.timeout,
                verify=self.verify_ssl,
                headers={
                    "Content-Type": "application/json;charset=utf-8",
                    "Accept": "application/json",
                    "X-XSRF-Protected": "1"
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"POST {endpoint} failed: {e}")
            return {"error": str(e)}

    def _execute_command(self, command: str, parameters: Optional[Dict] = None) -> Dict:
        """
        Execute OSC command via /osc/commands/execute.

        Args:
            command: OSC command name (e.g., "camera.startCapture").
            parameters: Command parameters.

        Returns:
            Response dict.
        """
        payload = {"name": command}
        if parameters:
            payload["parameters"] = parameters

        result = self._post("/osc/commands/execute", payload)
        if "error" in result:
            logger.error(f"OSC command '{command}' failed: {result['error']}")
        return result

    # ==================== Info & State APIs ====================

    def get_info(self) -> Dict:
        """
        Get camera information (/osc/info).

        Returns:
            Dict with manufacturer, model, serialNumber, firmwareVersion,
            gps, gyro, uptime, endpoints, api, apiLevel, cameraId, etc.
        """
        return self._get("/osc/info")

    def get_state(self) -> Dict:
        """
        Get camera state (/osc/state).

        Returns:
            Dict with fingerprint and state (batteryLevel, storageUri,
            temperatures, connectivity status, etc.).
        """
        return self._get("/osc/state")

    def get_camera_state(self) -> Optional[CameraState]:
        """
        Get structured camera state.

        Returns:
            CameraState object or None if failed.
        """
        data = self.get_state()
        if "error" in data or "state" not in data:
            return None

        state = data.get("state", {})
        return CameraState(
            fingerprint=data.get("fingerprint", ""),
            battery_level=state.get("batteryLevel", -1),
            storage_uri=state.get("storageUri", ""),
            gps_status=state.get("_gpsStatus", False),
            wifi_status=state.get("_wifiStatus", False),
            ethernet_status=state.get("_ethernetStatus", False),
            cellular_status=state.get("_cellularStatus", False),
            temperature_5g_c=state.get("_temperature5gC", -999),
            temperature_battery_c=state.get("_temperatureBattC", -999),
            temperature_sensor_min_c=state.get("_temperatureSensorMinC", -999),
            temperature_sensor_max_c=state.get("_temperatureSensorMaxC", -999),
            oos_status=state.get("_oosStatus", "unknown")
        )

    def get_battery_status(self) -> Optional[BatteryStatus]:
        """
        Get battery status.

        Returns:
            BatteryStatus object or None if failed.
        """
        state = self.get_camera_state()
        if not state:
            return None

        return BatteryStatus(
            level=state.battery_level,
            temperature_c=state.temperature_battery_c,
            is_present=state.battery_level >= 0
        )

    def check_for_updates(self) -> Dict:
        """
        Check for state updates (/osc/checkForUpdates).

        Returns:
            Dict with stateFingerprint and throttleTimeout.
        """
        return self._get("/osc/checkForUpdates")

    def get_command_status(self) -> Dict:
        """
        Get status of previous commands (/osc/commands/status).

        Returns:
            Dict with command name, id, state, and progress.
        """
        return self._get("/osc/commands/status")

    def get_status(self) -> Dict:
        """
        Get comprehensive camera status (info + state combined).

        Returns:
            Dict with info, state, connected flag, and streaming flag.
        """
        info = self.get_info()
        state = self.get_state()

        return {
            "info": info,
            "state": state,
            "connected": "error" not in info and bool(info.get("model")),
            "battery_level": state.get("state", {}).get("batteryLevel", -1),
            "ethernet_connected": state.get("state", {}).get("_ethernetStatus", False),
            "wifi_connected": state.get("state", {}).get("_wifiStatus", False),
            "cellular_connected": state.get("state", {}).get("_cellularStatus", False),
        }

    # ==================== Capture Control ====================

    def start_capture(self) -> Dict:
        """
        Start capture (image interval or video based on captureMode).

        Returns:
            Response dict with state and progress.
        """
        result = self._execute_command("camera.startCapture")
        if "error" not in result:
            logger.info("Camera capture started")
        return result

    def stop_capture(self) -> Dict:
        """
        Stop capture/recording.

        Returns:
            Response dict with captured file URLs.
        """
        result = self._execute_command("camera.stopCapture")
        if "error" not in result:
            logger.info("Camera capture stopped")
        return result

    def take_picture(self) -> Dict:
        """
        Capture a single equirectangular image.

        Note: Not available if _blurFaces or _blurLicensePlates is enabled.

        Returns:
            Response dict with fileUri of captured image.
        """
        result = self._execute_command("camera.takePicture")
        if "error" not in result:
            file_url = result.get("result", {}).get("fileUri", "")
            logger.info(f"Picture captured: {file_url}")
        return result

    # ==================== Live Streaming Control ====================

    def start_live(self) -> Dict:
        """
        Start RTSP/SRT live streaming.

        Note: Streaming protocol must also be enabled in camera WebUI.
        If video pipeline was stopped, this restarts it (~25s).

        Returns:
            Response dict.
        """
        result = self._execute_command("camera._startLive")
        if "error" not in result:
            logger.info("Live streaming started")
        return result

    def stop_live(self) -> Dict:
        """
        Stop RTSP/SRT live streaming.

        Warning: Stops the entire video pipeline. Use start_live to restart.

        Returns:
            Response dict.
        """
        result = self._execute_command("camera._stopLive")
        if "error" not in result:
            logger.info("Live streaming stopped")
        return result

    # ==================== File Management ====================

    def list_files(self) -> List[Dict]:
        """
        List available recordings and captured images.

        Returns:
            List of file entries with name, fileUrl, size, dateTimeZone.
        """
        result = self._execute_command("camera.listFiles")
        return result.get("results", {}).get("entries", [])

    def delete_file(self, file_url: str) -> bool:
        """
        Delete a file from camera storage.

        Args:
            file_url: URL of file to delete.

        Returns:
            True if successful.
        """
        result = self._execute_command("camera.delete", {"fileUri": file_url})
        return result.get("state") == "done"

    # ==================== Camera Options ====================

    def get_options(self, option_names: List[str]) -> Dict:
        """
        Get camera options/settings.

        Args:
            option_names: List of option names to retrieve.

        Returns:
            Dict of option values.
        """
        result = self._execute_command(
            "camera.getOptions",
            {"optionNames": option_names}
        )
        return result.get("results", {}).get("options", {})

    def set_options(self, options: Dict) -> bool:
        """
        Set camera options/settings.

        Args:
            options: Dict of option name to value.

        Returns:
            True if successful.
        """
        result = self._execute_command("camera.setOptions", {"options": options})
        success = result.get("state") == "done"
        if success:
            logger.info(f"Camera options updated: {list(options.keys())}")
        return success

    # ==================== Video Settings ====================

    def get_video_settings(self) -> Dict:
        """
        Get current video settings.

        Returns:
            Dict with bitrate, resolution, framerate, codec settings.
        """
        return self.get_options([
            "_bitRate", "_resolution", "_framerate", "_hevc",
            "_rateControl", "hdr", "_streamingProtocol"
        ])

    def set_video_settings(
        self,
        bitrate: Optional[int] = None,
        resolution: Optional[str] = None,
        framerate: Optional[int] = None,
        codec: Optional[str] = None,
        hdr: Optional[bool] = None,
        rate_control: Optional[str] = None
    ) -> bool:
        """
        Set video streaming settings.

        Args:
            bitrate: Bitrate in Mbps (8-100).
            resolution: "8k", "6k", or "4k".
            framerate: Frame rate (2-30).
            codec: "h264" or "h265".
            hdr: Enable HDR (only with 30fps, 4k/6k/8k).
            rate_control: "vbr" or "cbr".

        Returns:
            True if successful.
        """
        options = {}

        if bitrate is not None:
            options["_bitRate"] = bitrate

        if resolution is not None:
            options["_resolution"] = resolution.lower()

        if framerate is not None:
            options["_framerate"] = framerate

        if codec is not None:
            options["_hevc"] = codec.lower() == "h265"

        if hdr is not None:
            options["hdr"] = "hdr" if hdr else "off"

        if rate_control is not None:
            options["_rateControl"] = rate_control.lower()

        if options:
            return self.set_options(options)
        return True

    def set_streaming_protocol(self, protocol: str) -> bool:
        """
        Set streaming protocol.

        Args:
            protocol: One of "rtsp", "srtServer", "srtClient",
                     "rtsp_srtServer", "rtsp_srtClient", "off".

        Returns:
            True if successful.
        """
        return self.set_options({"_streamingProtocol": protocol})

    # ==================== Audio Settings ====================

    def get_audio_config(self) -> Dict:
        """
        Get audio/microphone configuration.

        Returns:
            Dict with _audio_enabled, _audioFormat, _volume_pct.
        """
        result = self._execute_command("camera._get_audio_config")
        return result.get("results", {}).get("options", {})

    def set_audio_config(
        self,
        enabled: Optional[bool] = None,
        audio_format: Optional[str] = None,
        volume_pct: Optional[int] = None
    ) -> bool:
        """
        Set audio/microphone configuration.

        Args:
            enabled: Enable/disable microphone.
            audio_format: "Ambisonics", "Stereo", or "Raw".
            volume_pct: Microphone volume (0-100).

        Returns:
            True if successful.
        """
        options = {}

        if enabled is not None:
            options["_audio_enabled"] = enabled

        if audio_format is not None:
            options["_audioFormat"] = audio_format

        if volume_pct is not None:
            options["_volume_pct"] = volume_pct

        if options:
            result = self._execute_command(
                "camera._set_audio_config",
                {"options": options}
            )
            return result.get("state") == "done"
        return True

    # ==================== Speaker Playback ====================

    def get_speaker_playback(self) -> Dict:
        """
        Get speaker playback settings.

        Returns:
            Dict with _files, _file_name, _play_type, _volume_pct.
        """
        result = self._execute_command("camera._get_speaker_playback")
        return result.get("results", {}).get("options", {})

    def set_speaker_playback(
        self,
        file_name: Optional[str] = None,
        play_type: Optional[str] = None,
        volume_pct: Optional[int] = None
    ) -> bool:
        """
        Set speaker playback settings.

        Args:
            file_name: Audio file name to play.
            play_type: "once", "license_plate_detected", "face_detected",
                      or "face_or_license_detected".
            volume_pct: Speaker volume (0-100).

        Returns:
            True if successful.
        """
        options = {}

        if file_name is not None:
            options["_file_name"] = file_name

        if play_type is not None:
            options["_play_type"] = play_type

        if volume_pct is not None:
            options["_volume_pct"] = volume_pct

        if options:
            result = self._execute_command(
                "camera._set_speaker_playback",
                {"options": options}
            )
            return result.get("state") == "done"
        return True

    def start_speaker_playback(self) -> bool:
        """
        Start playing audio file according to speaker settings.

        Returns:
            True if successful.
        """
        result = self._execute_command("camera._start_speaker_playback")
        return result.get("state") == "done"

    # ==================== AI Features ====================

    def set_ai_detection(self, enabled: bool) -> bool:
        """
        Enable/disable AI detection.

        Args:
            enabled: Enable AI detection.

        Returns:
            True if successful.
        """
        return self.set_options({"_artificialIntelligence": enabled})

    def set_face_blur(self, enabled: bool) -> bool:
        """
        Enable/disable face blurring.

        Note: Supports up to 32 simultaneous face blurs.

        Args:
            enabled: Enable face blurring.

        Returns:
            True if successful.
        """
        return self.set_options({"_blurFaces": enabled})

    def set_license_plate_blur(self, enabled: bool) -> bool:
        """
        Enable/disable license plate blurring.

        Note: Supports up to 32 simultaneous license plate blurs.

        Args:
            enabled: Enable license plate blurring.

        Returns:
            True if successful.
        """
        return self.set_options({"_blurLicensePlates": enabled})

    def set_ir_mode(self, enabled: bool) -> bool:
        """
        Enable/disable night vision infrared mode.

        Args:
            enabled: Enable IR mode.

        Returns:
            True if successful.
        """
        return self.set_options({"_IrMode": enabled})

    # ==================== System Control ====================

    def reset(self) -> bool:
        """
        Reset OSC-specific configurations to defaults.

        Returns:
            True if successful.
        """
        result = self._execute_command("camera.reset")
        if result.get("state") == "done":
            logger.info("Camera settings reset to defaults")
            return True
        return False

    def reboot(self) -> bool:
        """
        Reboot the camera.

        Returns:
            True if reboot initiated.
        """
        result = self._execute_command("camera._reboot")
        if result.get("state") == "done":
            logger.info("Camera reboot initiated")
            return True
        return False

    def shutdown(self) -> bool:
        """
        Shutdown the camera (requires unplugging Ethernet after).

        Returns:
            True if shutdown initiated.
        """
        result = self._execute_command("camera._shutdown")
        if result.get("state") == "done":
            logger.warning("Camera shutdown initiated - unplug Ethernet to complete")
            return True
        return False

    # ==================== Connectivity Info ====================

    def get_5g_info(self) -> Dict:
        """
        Get 5G cellular connection information.

        Returns:
            Dict with mcc, mnc, tac, cid, ip, bands, rsrp, etc.
        """
        return self.get_options(["_5gInfo"])

    def get_wifi_info(self) -> Dict:
        """
        Get WiFi connection information.

        Returns:
            Dict with ssid, ip, rssi, linkspeed, channel, frequency.
        """
        return self.get_options(["_wifiInfo"])

    def get_gps_info(self) -> Dict:
        """
        Get GPS information.

        Returns:
            Dict with lat and lng in decimal degrees.
        """
        return self.get_options(["gpsInfo"])

    def set_gps_info(self, lat: float, lng: float) -> bool:
        """
        Manually set GPS coordinates.

        Args:
            lat: Latitude (-90 to 90).
            lng: Longitude (-180 to 180).

        Returns:
            True if successful.
        """
        return self.set_options({"gpsInfo": {"lat": lat, "lng": lng}})

    # ==================== Storage Info ====================

    def get_storage_info(self) -> Dict:
        """
        Get storage information.

        Returns:
            Dict with totalSpace and remainingSpace in bytes.
        """
        return self.get_options(["totalSpace", "remainingSpace"])

    # ==================== Utility Methods ====================

    def health_check(self) -> bool:
        """
        Check if camera is reachable and responding.

        Returns:
            True if camera is responding.
        """
        try:
            info = self.get_info()
            return "error" not in info and bool(info.get("model"))
        except Exception:
            return False

    @classmethod
    def from_env(cls) -> "CameraOSCClient":
        """
        Create client from environment variables.

        Environment variables:
            CAMERA_OSC_HOST: Camera IP address
            CAMERA_OSC_PORT: API port (default 443)
            CAMERA_OSC_USERNAME: API username (default Nokia360)
            CAMERA_OSC_PASSWORD: API password (default Nokia360)
            CAMERA_OSC_USE_HTTPS: Use HTTPS (default true)

        Returns:
            Configured CameraOSCClient instance.
        """
        return cls(
            host=os.getenv("CAMERA_OSC_HOST", "192.168.18.19"),
            port=int(os.getenv("CAMERA_OSC_PORT", "443")),
            username=os.getenv("CAMERA_OSC_USERNAME", cls.DEFAULT_USERNAME),
            password=os.getenv("CAMERA_OSC_PASSWORD", cls.DEFAULT_PASSWORD),
            use_https=os.getenv("CAMERA_OSC_USE_HTTPS", "true").lower() == "true"
        )

    def __repr__(self) -> str:
        return f"CameraOSCClient({self.host}:{self.port})"


# Convenience function for quick status check
def get_camera_battery(host: str = None) -> Optional[BatteryStatus]:
    """
    Quick helper to get camera battery status.

    Args:
        host: Camera IP (uses env var if not specified).

    Returns:
        BatteryStatus or None.
    """
    if host:
        client = CameraOSCClient(host)
    else:
        client = CameraOSCClient.from_env()
    return client.get_battery_status()
