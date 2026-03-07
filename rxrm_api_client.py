#!/usr/bin/env python3
"""
RXRM API Client for Nokia 360 AI Safety System.

Handles all communication with Nokia RXRM platform:
- JWT Authentication
- Device discovery
- Viewport/Broadcast creation and management
- Stream URL retrieval
- Health checks

Based on Nokia RXRM Developer Guide rel 25R1 SP1
"""

import time
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RXRMDevice:
    """Represents an RXRM device (camera)."""
    id: str
    name: str
    type: str
    status: str
    capabilities: Dict = field(default_factory=dict)


@dataclass
class RXRMViewport:
    """Represents an RXRM viewport."""
    id: str
    name: str
    device_id: str
    azimuth: float
    elevation: float
    hfov: float
    vfov: float = 45.0
    status: str = "created"


@dataclass
class RXRMBroadcast:
    """Represents an RXRM broadcast (published stream)."""
    id: str
    viewport_id: str
    protocol: str
    access_url: str
    resolution: Tuple[int, int]
    bitrate_kbps: int
    status: str = "active"
    stream_id: Optional[str] = None


class RXRMClient:
    """
    Client for Nokia RXRM API.
    
    Handles authentication, device management, viewport control,
    and stream publishing for the Nokia 360 AI Safety System.
    
    Example:
        >>> client = RXRMClient("http://192.168.1.100:3000", "admin", "password")
        >>> client.login()
        >>> devices = client.list_devices()
        >>> broadcast = client.create_broadcast(devices[0].id, viewport_config)
    """
    
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        rtsp_port: int = 8554
    ):
        """
        Initialize RXRM client.

        Args:
            base_url: RXRM server URL (e.g., http://192.168.1.100:3000)
            username: API username
            password: API password
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            rtsp_port: RTSP streaming port (default 8554)
        """
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rtsp_port = rtsp_port

        self.session = requests.Session()
        self.access_token: Optional[str] = None
        self.token_expiry: float = 0

        # Cache for resources
        self._devices_cache: Dict[str, RXRMDevice] = {}
        self._viewports_cache: Dict[str, RXRMViewport] = {}
        self._broadcasts_cache: Dict[str, RXRMBroadcast] = {}

        # Extract host for RTSP URL construction
        self._rtsp_host = self.base_url.split('//')[1].split(':')[0]

        logger.info(f"RXRM client initialized: {base_url} (RTSP port: {rtsp_port})")

    def _build_rtsp_url(self, broadcast_id: str) -> str:
        """
        Build RTSP URL using our configured host.

        The RXRM API may return access_url with a stale/wrong IP address
        if the server was moved. This method ensures we always use the
        IP from our base_url configuration.

        Args:
            broadcast_id: The broadcast UUID

        Returns:
            RTSP URL with correct host: rtsp://<host>:<port>/streams/<id>
        """
        return f"rtsp://{self._rtsp_host}:{self.rtsp_port}/streams/{broadcast_id}"

    # ==================== AUTHENTICATION ====================
    
    def login(self) -> bool:
        """
        Authenticate with RXRM and obtain access token.
        
        Returns:
            True if authentication successful, False otherwise.
            
        Raises:
            ConnectionError: If unable to connect to RXRM server.
        """
        logger.info("Authenticating with RXRM...")
        
        url = f"{self.base_url}/api/v1/security/login"
        payload = {
            "username": self.username,
            "password": self.password,
            "provider": "db"
        }
        
        try:
            response = self._request("POST", url, json=payload, auth_required=False)
            
            if response and 'access_token' in response:
                self.access_token = response['access_token']
                # Token typically valid for 1 hour, refresh at 50 minutes
                self.token_expiry = time.time() + 3000
                
                # Update session headers
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}'
                })
                
                logger.info("✓ RXRM authentication successful")
                return True
            else:
                logger.error("Authentication failed: No access token in response")
                return False
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def _ensure_authenticated(self):
        """Ensure we have a valid authentication token."""
        if self.access_token is None or time.time() >= self.token_expiry:
            if not self.login():
                raise RuntimeError("Failed to authenticate with RXRM")
    
    def logout(self):
        """Clear authentication and cleanup."""
        self.access_token = None
        self.token_expiry = 0
        self.session.headers.pop('Authorization', None)
        logger.info("Logged out from RXRM")
    
    # ==================== HTTP REQUEST HELPER ====================
    
    def _request(
        self,
        method: str,
        url: str,
        auth_required: bool = True,
        **kwargs
    ) -> Optional[Dict]:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            url: Request URL
            auth_required: Whether authentication is required
            **kwargs: Additional request arguments
            
        Returns:
            Response JSON or None on failure.
        """
        if auth_required:
            self._ensure_authenticated()
        
        kwargs.setdefault('timeout', self.timeout)
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                
                if response.status_code == 401 and auth_required:
                    # Token expired, re-authenticate
                    logger.warning("Token expired, re-authenticating...")
                    self.access_token = None
                    self._ensure_authenticated()
                    continue
                
                response.raise_for_status()
                
                if response.content:
                    return response.json()
                return {}
                
            except requests.exceptions.RequestException as e:
                error_details = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_body = e.response.json()
                        error_details = f"{e} - Response: {error_body}"
                    except:
                        try:
                            error_details = f"{e} - Response: {e.response.text}"
                        except:
                            pass
                
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {error_details}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts")
                    raise
        
        return None
    
    # ==================== DEVICE MANAGEMENT ====================
    
    def list_devices(self, refresh: bool = False) -> List[RXRMDevice]:
        """
        List all devices (cameras) connected to RXRM.
        
        Args:
            refresh: Force refresh from server instead of using cache.
            
        Returns:
            List of RXRMDevice objects.
        """
        if self._devices_cache and not refresh:
            return list(self._devices_cache.values())
        
        logger.info("Fetching device list from RXRM...")
        
        url = f"{self.base_url}/api/v1/devices"
        response = self._request("GET", url)
        
        devices = []
        if response:
            items = response if isinstance(response, list) else response.get('items', [])
            for item in items:
                device = RXRMDevice(
                    id=item.get('id', ''),
                    name=item.get('name', ''),
                    type=item.get('type', ''),
                    status=item.get('status', 'unknown'),
                    capabilities=item.get('capabilities', {})
                )
                devices.append(device)
                self._devices_cache[device.id] = device
        
        logger.info(f"Found {len(devices)} devices")
        return devices
    
    def get_device(self, device_id: str) -> Optional[RXRMDevice]:
        """Get a specific device by ID."""
        if device_id in self._devices_cache:
            return self._devices_cache[device_id]
        
        url = f"{self.base_url}/api/v1/devices/{device_id}"
        response = self._request("GET", url)
        
        if response:
            device = RXRMDevice(
                id=response.get('id', ''),
                name=response.get('name', ''),
                type=response.get('type', ''),
                status=response.get('status', 'unknown'),
                capabilities=response.get('capabilities', {})
            )
            self._devices_cache[device.id] = device
            return device
        
        return None
    
    def find_camera(self, name_pattern: str = "Nokia") -> Optional[RXRMDevice]:
        """
        Find a camera by name pattern.
        
        Args:
            name_pattern: Pattern to match in device name (case-insensitive).
            
        Returns:
            First matching device or None.
        """
        devices = self.list_devices()
        
        for device in devices:
            if name_pattern.lower() in device.name.lower():
                logger.info(f"Found camera: {device.name} (ID: {device.id})")
                return device
        
        logger.warning(f"No camera found matching pattern: {name_pattern}")
        return None
    
    # ==================== BROADCAST MANAGEMENT ====================
    
    def create_broadcast(
        self,
        device_id: str,
        viewport_config: Dict[str, Any],
        name: str = "",
        codec: str = "h264",
        resolution: Tuple[int, int] = (960, 544),
        bitrate_kbps: int = 5500,
        framerate: int = 30
    ) -> RXRMBroadcast:
        """
        Create a broadcast stream with viewport configuration.
        
        Args:
            device_id: Camera device ID.
            viewport_config: Dict with azimuth, elevation, hfov, vfov.
            name: Broadcast name.
            codec: Video codec (h264, h265).
            resolution: Resolution tuple (width, height).
            bitrate_kbps: Bitrate in kbps.
            framerate: Frame rate.
            
        Returns:
            RXRMBroadcast with access URL.
            
        Raises:
            RuntimeError: If broadcast creation fails.
        """
        azimuth = viewport_config.get('azimuth', 0)
        elevation = viewport_config.get('elevation', 0)
        hfov = viewport_config.get('hfov', 60)
        
        logger.info(f"Creating broadcast: {name or 'unnamed'} (azimuth={azimuth}, elevation={elevation}, hfov={hfov})")
        
        url = f"{self.base_url}/api/v1/broadcasts"
        
        # Build viewport object per RXRM API
        viewport = {
            "azimuth": azimuth,
            "elevation": elevation,
            "scale": hfov / 90.0,  # Normalize FOV to scale
            "rotation": 0
        }
        
        # Build payload
        payload = {
            "device_id": device_id,
            "video_codec": codec,
            "resolution": [resolution[0], resolution[1]],
            "kbitrate": bitrate_kbps,
            "publish_target_name": "RXRM Media Gateway",
            "active": True,
            "viewport": viewport
        }
        
        if name:
            payload["name"] = name
        
        response = self._request("POST", url, json=payload)
        
        if response:
            broadcast_id = response.get('id', '')
            stream_id = response.get('stream_id', '')
            access_url = self._build_rtsp_url(broadcast_id)

            broadcast = RXRMBroadcast(
                id=broadcast_id,
                viewport_id=broadcast_id,
                protocol="rtsp",
                access_url=access_url,
                resolution=resolution,
                bitrate_kbps=bitrate_kbps,
                status="active",
                stream_id=stream_id
            )

            self._broadcasts_cache[broadcast.id] = broadcast
            logger.info(f"✓ Broadcast created: {access_url}")
            return broadcast

        raise RuntimeError(f"Failed to create broadcast: {name}")
    
    def delete_broadcast(self, broadcast_id: str) -> bool:
        """
        Delete a broadcast.
        
        Args:
            broadcast_id: Broadcast ID to delete.
            
        Returns:
            True if successful.
        """
        url = f"{self.base_url}/api/v1/broadcasts/{broadcast_id}"
        
        try:
            self._request("DELETE", url)
            
            if broadcast_id in self._broadcasts_cache:
                del self._broadcasts_cache[broadcast_id]
            
            logger.info(f"Broadcast deleted: {broadcast_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete broadcast {broadcast_id}: {e}")
            return False
    
    def get_stream_url(self, broadcast_id: str) -> Optional[str]:
        """
        Get the RTSP stream URL for a broadcast.

        Always constructs URL using our configured host to handle IP changes.

        Args:
            broadcast_id: Broadcast ID.

        Returns:
            RTSP URL string or None if broadcast doesn't exist.
        """
        # Check cache first
        if broadcast_id in self._broadcasts_cache:
            return self._broadcasts_cache[broadcast_id].access_url

        # Verify broadcast exists on server
        url = f"{self.base_url}/api/v1/broadcasts/{broadcast_id}"
        response = self._request("GET", url)

        if response and response.get('id'):
            # Broadcast exists, build URL with our configured host
            return self._build_rtsp_url(broadcast_id)

        return None
    
    def update_viewport(
        self,
        broadcast_id: str,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        hfov: Optional[float] = None
    ) -> bool:
        """
        Update viewport parameters for an active broadcast.
        
        Args:
            broadcast_id: Broadcast ID.
            azimuth: New horizontal angle (optional).
            elevation: New vertical angle (optional).
            hfov: New field of view (optional).
            
        Returns:
            True if successful.
        """
        # Get the stream_id from the broadcast
        broadcast_url = f"{self.base_url}/api/v1/broadcasts/{broadcast_id}"
        
        try:
            broadcast_data = self._request("GET", broadcast_url)
            if not broadcast_data:
                logger.error(f"Failed to get broadcast {broadcast_id}")
                return False
            
            stream_id = broadcast_data.get('stream_id')
            if not stream_id:
                logger.error(f"Broadcast {broadcast_id} has no stream_id")
                return False
            
            # Update viewport using /viewport endpoint
            viewport_url = f"{self.base_url}/api/v1/viewport"
            
            payload = {
                "stream_id": stream_id,
                "rotation": 0
            }
            
            if azimuth is not None:
                payload['azimuth'] = float(azimuth)
            if elevation is not None:
                payload['elevation'] = float(elevation)
            if hfov is not None:
                payload['scale'] = float(hfov) / 90.0
            
            response = self._request("POST", viewport_url, json=payload)
            
            if response is not None:
                logger.debug(f"✓ Updated viewport for stream {stream_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update viewport: {e}")
            return False
    
    def list_broadcasts(self) -> List[RXRMBroadcast]:
        """
        List all active broadcasts.

        Note: RTSP URLs are constructed using our configured host to handle
        IP address changes. The RXRM API may return stale IPs in access_url.
        """
        url = f"{self.base_url}/api/v1/broadcasts"
        response = self._request("GET", url)

        broadcasts = []
        if response:
            items = response if isinstance(response, list) else response.get('items', [])
            for item in items:
                broadcast_id = item.get('id', '')
                # Build URL with our configured host (ignore API's access_url)
                access_url = self._build_rtsp_url(broadcast_id) if broadcast_id else ''

                broadcast = RXRMBroadcast(
                    id=broadcast_id,
                    viewport_id=item.get('viewportId', ''),
                    protocol=item.get('protocol', 'rtsp'),
                    access_url=access_url,
                    resolution=(item.get('width', 1920), item.get('height', 1080)),
                    bitrate_kbps=item.get('bitrate', 5500),
                    status=item.get('status', 'unknown'),
                    stream_id=item.get('stream_id')
                )
                broadcasts.append(broadcast)
                self._broadcasts_cache[broadcast.id] = broadcast

        return broadcasts

    def get_viewport_streams(self) -> Dict[str, str]:
        """
        Get RTSP URLs for all 4 viewports (front, right, back, left).

        This is a convenience method for auto-configuration.
        First tries broadcasts endpoint, then falls back to device streams.

        Returns:
            Dictionary mapping viewport names to RTSP URLs.
            Example: {"front": "rtsp://...", "right": "rtsp://...", ...}
        """
        viewport_streams = {}
        viewport_names = ['front', 'right', 'back', 'left']

        # Try method 1: Broadcasts endpoint (if streams are published)
        broadcasts = self.list_broadcasts()
        for broadcast in broadcasts:
            name = broadcast.id.lower() if broadcast.id else ''
            access_url = broadcast.access_url

            # Match broadcast name to viewport
            for viewport in viewport_names:
                if viewport in name and viewport not in viewport_streams:
                    viewport_streams[viewport] = access_url
                    logger.info(f"Found {viewport} viewport from broadcasts: {access_url}")
                    break

        # Log results
        missing = set(viewport_names) - set(viewport_streams.keys())
        if missing:
            logger.warning(f"Missing viewports: {missing}")
        else:
            logger.info(f"Found all 4 viewports")

        return viewport_streams

    # ==================== HEALTH & UTILITY ====================

    def health_check(self) -> bool:
        """
        Check if RXRM is reachable and responsive.

        This method checks health by attempting to list devices,
        which validates both connectivity and authentication.

        Returns:
            True if RXRM is healthy and accessible.
        """
        try:
            # Try to get devices - this validates auth and connectivity
            devices = self.list_devices(refresh=True)
            return len(devices) >= 0  # Even 0 devices is "healthy"
        except Exception as e:
            logger.error(f"RXRM health check failed: {e}")
            return False

    def get_server_info(self) -> Dict:
        """
        Get RXRM server information.

        Returns:
            Server info dict with version, capabilities, etc.
        """
        try:
            url = f"{self.base_url}/api/v1/info"
            response = self._request("GET", url)
            return response or {}
        except:
            return {}
    
    def cleanup(self):
        """Clean up all created broadcasts and viewports."""
        logger.info("Cleaning up RXRM resources...")
        
        # Delete all broadcasts
        for broadcast_id in list(self._broadcasts_cache.keys()):
            self.delete_broadcast(broadcast_id)
        
        logger.info("✓ RXRM cleanup complete")
    
    def get_broadcasts_cache(self) -> Dict[str, RXRMBroadcast]:
        """Get cached broadcasts."""
        return self._broadcasts_cache.copy()
