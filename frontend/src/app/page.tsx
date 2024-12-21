"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.css";
import msgpack from 'msgpack-lite';
import { WavRecorder, WavStreamPlayer } from 'wavtools';

export default function Home() {
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(true); // true means listening, false means speaking
  const [isPlayingAudio, setIsPlayingAudio] = useState(false); // State to track audio playback
  const [socket, setSocket] = useState<WebSocket | null>(null);

  // 定义可能的连接状态类型
  type ConnectionStatus = "Connecting..." | "Connected" | "Disconnected" | "Closed";

  const [connectionStatus, setConnectionStatus] = useState<string>("Connecting..."); // State to track connection status

  const SOCKET_URL = "wss://audio.enty.services/stream-vc";

  const [wavStreamPlayer] = useState(new WavStreamPlayer({ sampleRate: 22500 }));

  // Connect to audio output
  useEffect(() => {
    const connectWavStreamPlayer = async () => {
      try {
        await wavStreamPlayer.connect();
        console.log("WavStreamPlayer connected");
      } catch (error) {
        console.error("Failed to connect WavStreamPlayer", error);
      }
    };

    connectWavStreamPlayer();
  }, [wavStreamPlayer]);

  // Initialize WebSocket and media devices
  useEffect(() => {
    let wakeLock: WakeLockSentinel | null = null;

    // Request screen wake lock to prevent the screen from going to sleep
    async function requestWakeLock() {
      try {
        wakeLock = await navigator.wakeLock.request("screen");
        console.log("Screen wake lock acquired");
      } catch (error) {
        console.error("Failed to acquire wake lock", error);
      }
    }

    requestWakeLock();

    return () => {
      if (wakeLock) {
        wakeLock.release().then(() => {
          console.log("Screen wake lock released");
        }).catch((error) => {
          console.error("Failed to release wake lock", error);
        });
      }
    };
  }, []);

  // Access the microphone and start recording
  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        setMediaRecorder(new MediaRecorder(stream));
      }).catch((error) => {
        console.error("Error accessing media devices.", error);
      });
    } else {
      console.error("Media devices API not supported.");
    }
  }, []);

  // Handle WebSocket connection and messaging
  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://www.WebRTC-Experiment.com/RecordRTC.js";
    script.onload = () => {
      const RecordRTC = (window as any).RecordRTC;
      const StereoAudioRecorder = (window as any).StereoAudioRecorder;

      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
          let websocket: WebSocket | null = null;

          const reconnectWebSocket = () => {
            if (websocket) websocket.close();
            websocket = new WebSocket(SOCKET_URL);
            setSocket(websocket);

            websocket.onopen = () => {
              console.log("client connected to websocket");
              setConnectionStatus("Connected");

              const recorder = new RecordRTC(stream, {
                type: 'audio',
                recorderType: StereoAudioRecorder,
                mimeType: 'audio/wav',
                timeSlice: 500,
                desiredSampRate: 16000,
                numberOfAudioChannels: 1,
                ondataavailable: (blob: Blob) => {
                  if (blob.size > 0) {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                      if (reader.result) {
                        const base64data = arrayBufferToBase64(reader.result as ArrayBuffer);

                        const message = {
                          event: "start",
                          request: {
                            audio: base64data,
                            latency: "normal",
                            format: "opus",
                            prosody: {
                              speed: 1.0,
                              volume: 0
                            },
                            reference_id: "c9cf4e49"
                          }
                        };
                        const encodedData = JSON.stringify(message);
                        if (websocket) {
                          websocket.send(encodedData);
                        } else {
                          console.error("WebSocket is null, cannot send data.");
                        }
                      } else {
                        console.error("FileReader result is null");
                      }
                    };
                    reader.readAsArrayBuffer(blob);
                  }
                }
              });

              recorder.startRecording();
            };

            websocket.onmessage = (event) => {
              setIsRecording(false);
              setIsPlayingAudio(true);
              const arr = Uint8Array.from(event.data, (m) => m.codePointAt(0));
              const bytes = new Int16Array(arr.buffer);
              wavStreamPlayer.add16BitPCM(bytes, "tracker_id");
            };

            websocket.onclose = () => {
              console.log("WebSocket connection closed...");
              setConnectionStatus("Reconnecting...");
              setTimeout(reconnectWebSocket, 5000);
            };

            websocket.onerror = (error) => {
              console.error("WebSocket error:", error);
              websocket?.close();
            };
          };
          console.log("client start connect to websocket");
          reconnectWebSocket();
        }).catch((error) => {
          console.error("Error with getUserMedia", error);
        });
      }
    };
    document.body.appendChild(script);

    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, []);

  // Handle media recorder pause/resume
  useEffect(() => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      if (isRecording) {
        mediaRecorder.resume();
      } else {
        mediaRecorder.pause();
      }
    }
  }, [isRecording, mediaRecorder]);

  function arrayBufferToBase64(arrayBuffer: ArrayBuffer): string {
    let binary = '';
    const uint8Array = new Uint8Array(arrayBuffer);
    const len = uint8Array.length;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(uint8Array[i]);
    }
    return btoa(binary);
  }

  const [isInCall, setIsInCall] = useState(true);

  function infiniteSleep(): Promise<void> {
    return new Promise(() => {}); // 不调用 resolve 或 reject
  }

  async function endCall() {
    if (socket) {
      socket.close();
      setSocket(null);
    }
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
      setMediaRecorder(null);
    }
    setIsInCall(false);
    setIsRecording(false);
    setIsPlayingAudio(false);
    setConnectionStatus("Closed");

    await infiniteSleep();
  }

  return (
    <div className={styles.container}>
      <div className={styles.statusBar}>
        <div className={styles.connectionStatus}>
          <div
            className={`${styles.statusDot} ${
              connectionStatus === "Connected" ? styles.connected : ""
            }`}
          />
          {connectionStatus}
        </div>
      </div>

      <div className={styles.mainContent}>
        <div className={styles.avatarSection}>
          <div
            className={`${styles.avatarContainer} ${
              isPlayingAudio ? styles.speaking : ""
            }`}
          >
            <img src="/ai-avatar.png" alt="AI" className={styles.avatar} />
          </div>
          <div className={styles.status}>
            <span
              className={
                isInCall
                  ? isPlayingAudio
                    ? styles.speakingAnimation
                    : styles.listeningAnimation
                  : styles.offlineAnimation
              }
            >
              {isInCall
                ? isPlayingAudio
                  ? "AI正在说话"
                  : "AI正在听"
                : "AI 离线"}
            </span>
          </div>
        </div>
      </div>

      <div className={styles.controls}>
        <button
          className={isInCall ? styles.endCallButton : styles.startCallButton}
          onClick={() => {
            if (isInCall) {
              endCall();
            } else {
              window.location.reload();
            }
          }}
        >
          {isInCall ? "结束通话" : "重新通话"}
        </button>
      </div>
    </div>
  );
}
