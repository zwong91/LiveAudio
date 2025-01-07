## 1. 配置

```

stun:108.136.246.72:3478

turn:108.136.246.72:3478
admin/7f0dd067662502af36934e85b43895b148edfcdb


stun:gtp.aleopool.cc:3478

turn:gtp.aleopool.cc:3478
admin/7f0dd067662502af36934e85b43895b148edfcdb


stun:audio.enty.services:3478

turn:audio.enty.services:3478
admin/7f0dd067662502af36934e85b43895b148edfcdb



turnutils_stunclient -p 3478 audio.enty.services

root@10-60-3-26:~/LiveAudio-rtc# curl -v telnet://audio.enty.services:3478
* Host audio.enty.services:3478 was resolved.
* IPv6: (none)
* IPv4: 108.137.9.108
*   Trying 108.137.9.108:3478...
* Connected to audio.enty.services (108.137.9.108) port 3478

turnutils_turnclient -p 3478 -u admin -w 7f0dd067662502af36934e85b43895b148edfcdb audio.enty.services

```

## 2. 测试
chrome://webrtc-internals
https://icetest.info
https://webrtc.github.io/samples/src/content/peerconnection/trickle-ice/


```
ubuntu@ip-172-31-9-141:~$ tcping audio.enty.services  3478
TCPinging audio.enty.services on port 3478
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=1 time=0.910 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=2 time=0.662 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=3 time=0.690 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=4 time=0.696 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=5 time=0.635 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=6 time=0.671 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=7 time=0.667 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=8 time=0.653 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=9 time=0.668 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=10 time=0.676 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=11 time=0.675 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=12 time=0.648 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=13 time=0.635 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=14 time=0.664 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=15 time=0.653 ms
Reply from audio.enty.services (108.137.9.108) on port 3478 TCP_conn=16 time=0.654 ms
^C
--- audio.enty.services (108.137.9.108) TCPing statistics ---
16 probes transmitted on port 3478 | 16 received, 0.00% packet loss
successful probes:   16
unsuccessful probes: 0
last successful probe:   2024-12-31 05:01:55
last unsuccessful probe: Never failed
total uptime:   16 seconds
total downtime: 0 second
longest consecutive uptime:   16 seconds from 2024-12-31 05:01:40 to 2024-12-31 05:01:56
retried to resolve hostname 0 times
rtt min/avg/max: 0.635/0.678/0.910 ms
--------------------------------------
TCPing started at: 2024-12-31 05:01:40
TCPing ended at:   2024-12-31 05:01:56
duration (HH:MM:SS): 00:00:16

```