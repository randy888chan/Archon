#!/usr/bin/env python3
"""
Enhanced Socket.IO Event Streaming Validation Test

This test validates the comprehensive Socket.IO event streaming improvements
including heartbeat events, performance metrics, stall detection, and 
granular progress updates with detailed operation information.

Features being tested:
- crawl_heartbeat events (every 5 seconds during processing)
- crawl_performance metrics streaming  
- crawl_stall_detected alerts
- crawl_recovery notifications
- Enhanced progress events with batch processing details
- Performance metrics and ETA calculations
- Error event streaming with diagnostic information

SUCCESS METRICS:
✅ Events stream continuously (not just at batch boundaries)
✅ Detailed operation information in each event
✅ Performance metrics update in real-time
✅ Error/recovery events provide clear information
"""

import asyncio
import json
import sys
import time
from typing import Any, Dict, List
import uuid
import aiohttp
import socketio

class SocketIOEventTester:
    """Test enhanced Socket.IO event streaming capabilities."""
    
    def __init__(self, server_url: str = "http://localhost:8181"):
        self.server_url = server_url
        self.sio = socketio.AsyncClient(
            logger=False,
            engineio_logger=False,
            reconnection=True,
            reconnection_attempts=3,
            reconnection_delay=1
        )
        self.events_received: List[Dict[str, Any]] = []
        self.event_counts: Dict[str, int] = {}
        self.test_results = {
            'heartbeat_events': False,
            'performance_events': False,
            'stall_detection': False,
            'recovery_events': False,
            'detailed_progress': False,
            'batch_processing': False,
            'error_streaming': False,
            'continuous_streaming': False
        }
        
    async def setup_event_handlers(self):
        """Setup handlers for all enhanced Socket.IO events."""
        
        @self.sio.event
        async def connect():
            print("🔌 Connected to Socket.IO server")
            
        @self.sio.event
        async def disconnect():
            print("🔌 Disconnected from Socket.IO server")
            
        @self.sio.event
        async def connect_error(data):
            print(f"❌ Connection error: {data}")
            
        # Enhanced crawl progress events
        @self.sio.event
        async def crawl_progress(data):
            await self.handle_event('crawl_progress', data)
            
        @self.sio.event  
        async def crawl_heartbeat(data):
            await self.handle_event('crawl_heartbeat', data)
            
        @self.sio.event
        async def crawl_performance(data):
            await self.handle_event('crawl_performance', data)
            
        @self.sio.event
        async def crawl_stall_detected(data):
            await self.handle_event('crawl_stall_detected', data)
            
        @self.sio.event
        async def crawl_recovery(data):
            await self.handle_event('crawl_recovery', data)
            
        # Error and status events
        @self.sio.event
        async def error(data):
            await self.handle_event('error', data)
            
        @self.sio.event
        async def crawl_subscribe_ack(data):
            await self.handle_event('crawl_subscribe_ack', data)
            print(f"✅ Subscription acknowledged: {data}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any]):
        """Handle and analyze received events."""
        timestamp = time.time()
        self.events_received.append({
            'event_type': event_type,
            'data': data,
            'timestamp': timestamp,
            'received_at': time.strftime('%H:%M:%S', time.localtime(timestamp))
        })
        
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
        
        # Analyze event for test criteria
        await self.analyze_event(event_type, data)
        
        # Print event details
        self.print_event_summary(event_type, data)
    
    async def analyze_event(self, event_type: str, data: Dict[str, Any]):
        """Analyze events against success criteria."""
        
        # Check for heartbeat events
        if event_type == 'crawl_heartbeat':
            self.test_results['heartbeat_events'] = True
            
        # Check for performance metrics
        if event_type == 'crawl_performance' or ('processing_rate' in data or 'items_processed' in data):
            self.test_results['performance_events'] = True
            
        # Check for stall detection
        if event_type == 'crawl_stall_detected' or data.get('status') == 'stalled':
            self.test_results['stall_detection'] = True
            
        # Check for recovery notifications
        if event_type == 'crawl_recovery' or data.get('type') == 'recovery':
            self.test_results['recovery_events'] = True
            
        # Check for detailed progress information
        if any(key in data for key in ['stage', 'substage', 'currentUrl', 'batchDetails']):
            self.test_results['detailed_progress'] = True
            
        # Check for batch processing details
        if 'batchDetails' in data or 'batch_size' in data:
            self.test_results['batch_processing'] = True
            
        # Check for error streaming
        if event_type == 'error' or 'error' in data:
            self.test_results['error_streaming'] = True
            
        # Check for continuous streaming (events coming frequently)
        if len(self.events_received) > 1:
            recent_events = [e for e in self.events_received if time.time() - e['timestamp'] < 10]
            if len(recent_events) > 5:  # More than 5 events in 10 seconds indicates continuous streaming
                self.test_results['continuous_streaming'] = True
    
    def print_event_summary(self, event_type: str, data: Dict[str, Any]):
        """Print formatted event information."""
        
        # Color coding for different event types
        colors = {
            'crawl_progress': '📊',
            'crawl_heartbeat': '💓',
            'crawl_performance': '⚡',
            'crawl_stall_detected': '⚠️',
            'crawl_recovery': '🔄',
            'error': '❌'
        }
        
        icon = colors.get(event_type, '📡')
        
        # Extract key information
        status = data.get('status', 'unknown')
        progress = data.get('percentage', data.get('progress', 'N/A'))
        stage = data.get('stage', '')
        substage = data.get('substage', '')
        processing_rate = data.get('processing_rate', 0)
        
        # Format stage info
        stage_info = stage
        if substage:
            stage_info += f".{substage}"
            
        # Format progress info
        progress_info = f"{progress}%" if isinstance(progress, (int, float)) else str(progress)
        
        # Format performance info
        perf_info = ""
        if processing_rate > 0:
            perf_info = f" | rate={processing_rate}/s"
        if data.get('items_processed') and data.get('total_items'):
            perf_info += f" | items={data['items_processed']}/{data['total_items']}"
            
        print(f"{icon} {event_type}: status={status} | progress={progress_info} | stage={stage_info}{perf_info}")
        
        # Print additional details for special events
        if event_type == 'crawl_heartbeat':
            print(f"   💓 Heartbeat: system actively processing")
        elif event_type == 'crawl_performance':
            print(f"   ⚡ Performance metrics: {json.dumps(data.get('metrics', {}), indent=2)}")
        elif event_type == 'crawl_stall_detected':
            print(f"   ⚠️  Stall detected: {data.get('reason', 'unknown')}")
        elif event_type == 'crawl_recovery':
            print(f"   🔄 Recovery: {data.get('recovery_method', 'unknown')}")
    
    async def connect_and_subscribe(self, progress_id: str):
        """Connect to Socket.IO and subscribe to progress updates."""
        try:
            print(f"🔗 Connecting to {self.server_url}...")
            await self.sio.connect(self.server_url, transports=['websocket', 'polling'])
            print("✅ Connected successfully")
            
            # Subscribe to crawl progress
            print(f"📡 Subscribing to progress updates for: {progress_id}")
            await self.sio.emit('crawl_subscribe', {'progress_id': progress_id})
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    async def start_test_crawl(self) -> str:
        """Start a test crawl to generate Socket.IO events."""
        try:
            async with aiohttp.ClientSession() as session:
                crawl_data = {
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "knowledge_type": "technical",
                    "tags": ["python", "asyncio", "documentation"],
                    "max_depth": 1,
                    "extract_code_examples": True
                }
                
                print(f"🚀 Starting test crawl for URL: {crawl_data['url']}")
                
                async with session.post(
                    f"{self.server_url}/api/knowledge-items/crawl", 
                    json=crawl_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        progress_id = result.get('progressId')
                        print(f"✅ Crawl started successfully. Progress ID: {progress_id}")
                        return progress_id
                    else:
                        text = await response.text()
                        print(f"❌ Failed to start crawl: {response.status} - {text}")
                        return None
                        
        except Exception as e:
            print(f"❌ Error starting crawl: {e}")
            return None
    
    async def monitor_events(self, duration: int = 60):
        """Monitor events for specified duration."""
        print(f"👀 Monitoring events for {duration} seconds...")
        
        start_time = time.time()
        last_summary = start_time
        
        while time.time() - start_time < duration:
            await asyncio.sleep(1)
            
            # Print summary every 10 seconds
            if time.time() - last_summary >= 10:
                self.print_monitoring_summary()
                last_summary = time.time()
        
        print("⏰ Monitoring period completed")
    
    def print_monitoring_summary(self):
        """Print current monitoring status."""
        total_events = len(self.events_received)
        recent_events = len([e for e in self.events_received if time.time() - e['timestamp'] < 10])
        
        print(f"\n📊 MONITORING SUMMARY:")
        print(f"   Total events: {total_events}")
        print(f"   Recent events (10s): {recent_events}")
        print(f"   Event types: {dict(self.event_counts)}")
        print(f"   Test progress: {sum(self.test_results.values())}/{len(self.test_results)} criteria met")
        print()
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("🧪 SOCKET.IO EVENT STREAMING VALIDATION REPORT")
        print("="*80)
        
        # Overall statistics
        total_events = len(self.events_received)
        duration = max((e['timestamp'] for e in self.events_received), default=0) - \
                  min((e['timestamp'] for e in self.events_received), default=0)
        event_rate = total_events / max(duration, 1)
        
        print(f"\n📊 EVENT STATISTICS:")
        print(f"   Total events received: {total_events}")
        print(f"   Monitoring duration: {duration:.1f} seconds") 
        print(f"   Average event rate: {event_rate:.2f} events/second")
        print(f"   Unique event types: {len(self.event_counts)}")
        
        print(f"\n📡 EVENT TYPE BREAKDOWN:")
        for event_type, count in sorted(self.event_counts.items()):
            percentage = (count / total_events * 100) if total_events > 0 else 0
            print(f"   {event_type}: {count} events ({percentage:.1f}%)")
        
        print(f"\n✅ SUCCESS CRITERIA VALIDATION:")
        criteria_met = 0
        total_criteria = len(self.test_results)
        
        for criterion, met in self.test_results.items():
            status = "✅ PASS" if met else "❌ FAIL"
            print(f"   {criterion}: {status}")
            if met:
                criteria_met += 1
        
        print(f"\n🎯 OVERALL RESULT:")
        success_rate = (criteria_met / total_criteria * 100)
        if success_rate >= 80:
            print(f"   🎉 SUCCESS: {criteria_met}/{total_criteria} criteria met ({success_rate:.1f}%)")
            print("   Enhanced Socket.IO event streaming is working correctly!")
        elif success_rate >= 60:
            print(f"   ⚠️  PARTIAL: {criteria_met}/{total_criteria} criteria met ({success_rate:.1f}%)")
            print("   Some enhanced features need attention.")
        else:
            print(f"   ❌ FAILURE: {criteria_met}/{total_criteria} criteria met ({success_rate:.1f}%)")
            print("   Enhanced event streaming needs significant improvements.")
        
        # Detailed event analysis
        if total_events > 0:
            print(f"\n📋 DETAILED EVENT ANALYSIS:")
            
            # Check continuous streaming
            if self.test_results['continuous_streaming']:
                print("   ✅ Events stream continuously (not sparse)")
            else:
                print("   ❌ Events appear sparse - need more frequent updates")
            
            # Check detailed information
            detailed_events = sum(1 for e in self.events_received 
                                if any(key in e['data'] for key in ['stage', 'substage', 'batchDetails']))
            if detailed_events > total_events * 0.5:
                print("   ✅ Events contain detailed operation information")
            else:
                print("   ❌ Events lack detailed operation information")
            
            # Check performance metrics
            perf_events = sum(1 for e in self.events_received 
                            if any(key in e['data'] for key in ['processing_rate', 'items_processed', 'eta']))
            if perf_events > 0:
                print("   ✅ Performance metrics are included")
            else:
                print("   ❌ Performance metrics missing")
        
        print(f"\n📄 SAMPLE EVENTS:")
        for event_type in ['crawl_heartbeat', 'crawl_performance', 'crawl_progress']:
            sample_events = [e for e in self.events_received if e['event_type'] == event_type][:2]
            if sample_events:
                print(f"\n   {event_type} samples:")
                for i, event in enumerate(sample_events, 1):
                    print(f"   {i}. {json.dumps(event['data'], indent=6)}")
        
        return success_rate >= 80
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.sio.connected:
            await self.sio.disconnect()

async def main():
    """Run the comprehensive Socket.IO event streaming test."""
    print("🧪 Starting Enhanced Socket.IO Event Streaming Validation")
    print("="*60)
    
    tester = SocketIOEventTester()
    
    try:
        # Setup event handlers
        await tester.setup_event_handlers()
        
        # Start a test crawl
        progress_id = await tester.start_test_crawl()
        if not progress_id:
            print("❌ Could not start test crawl - exiting")
            return False
        
        # Connect and subscribe to events
        connected = await tester.connect_and_subscribe(progress_id)
        if not connected:
            print("❌ Could not connect to Socket.IO - exiting")
            return False
        
        # Monitor events for sufficient duration
        await tester.monitor_events(duration=45)  # 45 seconds should be enough
        
        # Generate comprehensive report
        success = tester.generate_test_report()
        
        return success
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    # Check if server URL provided
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8181"
    
    try:
        # Run the test
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Failed to run test: {e}")
        sys.exit(1)