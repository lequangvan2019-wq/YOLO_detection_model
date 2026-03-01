import json
import ssl
import time
import sys

# ═══════════════════════════════════════════════════════════════════════════
# LED OUTPUT MODULE - Simple 0/1 Status Publisher
# ═══════════════════════════════════════════════════════════════════════════

print("="*60)
print("LED Test - Safety Vest Detection Status Publisher")
print("="*60)

# MQTT Configuration
MQTT_HOST = "a3eb353511ab48eda8f1365c60928f57.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "lequangvan2003"
MQTT_PASSWORD = "Cockeo20062003%"
MQTT_TOPIC_LED = "safetyvest/led"  # Output: 0 or 1
MQTT_TOPIC_INPUT = "safetyvest/status"  # Listen to main detection system
MQTT_PUBLISH_INTERVAL_SEC = 1.0

# Check MQTT availability
try:
    import paho.mqtt.client as mqtt
    print("✓ paho-mqtt: OK")
    MQTT_AVAILABLE = True
except ImportError as e:
    print(f"✗ paho-mqtt: {e}")
    print("  Run: pip install paho-mqtt")
    MQTT_AVAILABLE = False
    sys.exit(1)


class LEDStatusPublisher:
    """
    Publishes LED status (0 or 1) based on vest detection.
    - 0: No people wearing safety vest
    - 1: At least 1 person wearing safety vest
    """
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.current_led_status = 0  # 0 = no vest, 1 = at least one vest
        self.last_people_count = 0
        self.last_vest_count = 0
        self.last_publish = 0.0
        
    def connect(self):
        """Connect to MQTT broker"""
        if not MQTT_AVAILABLE:
            print("✗ MQTT not available")
            return False
        
        try:
            self.client = mqtt.Client(protocol=mqtt.MQTTv311)
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
            self.client.tls_insecure_set(False)
            
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            
            self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            self.client.subscribe(MQTT_TOPIC_INPUT)  # Listen to detection results
            self.client.loop_start()
            
            # Wait a moment for connection
            timeout = time.time() + 5
            while not self.connected and time.time() < timeout:
                time.sleep(0.1)
            
            if self.connected:
                print(f"✓ Connected to MQTT broker")
                print(f"✓ Listening on: {MQTT_TOPIC_INPUT}")
                print(f"✓ Publishing LED status to: {MQTT_TOPIC_LED}")
                return True
            else:
                print("✗ Failed to connect to MQTT broker (timeout)")
                return False
                
        except Exception as e:
            print(f"✗ MQTT connection error: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        self.connected = (rc == 0)
        if self.connected:
            print("✓ MQTT connection established")
        else:
            print(f"? MQTT connection failed with code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.connected = False
        if rc != 0:
            print(f"⚠ Unexpected MQTT disconnection: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Handle incoming messages from main detection system"""
        try:
            payload = json.loads(msg.payload.decode())
            
            current_with_vest = payload.get("current_with_vest", 0)
            current_persons = payload.get("current_persons", 0)
            
            # Update LED status
            # 1 if at least one person is wearing a vest
            # 0 if no one is wearing a vest
            self.current_led_status = 1 if current_with_vest > 0 else 0
            
            # Store counts for reference
            self.last_people_count = current_persons
            self.last_vest_count = current_with_vest
            
            # Publish LED status
            self.publish_led_status()
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def publish_led_status(self):
        """Publish LED status (0 or 1) to MQTT"""
        now = time.time()
        if now - self.last_publish < MQTT_PUBLISH_INTERVAL_SEC:
            return
        
        if not self.client or not self.connected:
            return
        
        try:
            # Simple payload - just "0" or "1"
            payload = "1" if self.current_led_status == 1 else "0"
            
            self.client.publish(
                MQTT_TOPIC_LED, 
                payload, 
                qos=1, 
                retain=False
            )
            
            self.last_publish = now
            
            # Print for debugging
            status_text = "✓ VEST DETECTED" if self.current_led_status == 1 else "✗ NO VEST"
            print(f"[LED={self.current_led_status}] {status_text} | People: {self.last_people_count}, Vest: {self.last_vest_count}")
            
        except Exception as e:
            print(f"Error publishing LED status: {e}")
    
    def close(self):
        """Disconnect from MQTT broker"""
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                print("✓ MQTT disconnected")
            except Exception:
                pass
            self.client = None


def main():
    """Main function to run LED publisher"""
    publisher = LEDStatusPublisher()
    
    if not publisher.connect():
        print("Failed to initialize LED publisher")
        return
    
    print("\n" + "="*60)
    print("LED Publisher running. Press Ctrl+C to exit.")
    print("="*60 + "\n")
    
    try:
        # Keep the publisher running
        while True:
            time.sleep(1)
            publisher.publish_led_status()
            
            # Check connection status
            if not publisher.connected:
                print("⚠ Connection lost. Attempting to reconnect...")
                if publisher.connect():
                    print("✓ Reconnected")
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        publisher.close()
        print("✓ LED Publisher stopped")


if __name__ == "__main__":
    main()
