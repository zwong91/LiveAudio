"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.css";
import { WavRecorder, WavStreamPlayer } from 'wavtools';

export default function Home() {
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(true); // true means listening, false means speaking
  const [isPlayingAudio, setIsPlayingAudio] = useState(false); // State to track audio playback
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
  const [audioDuration, setAudioDuration] = useState<number>(0); // State to track audio duration

  // Define possible connection statuses
  type ConnectionStatus = "Connecting..." | "Connected" | "Disconnected" | "Closed";
  const [connectionStatus, setConnectionStatus] = useState<string>("Connecting..."); // State to track connection status

  const [isInCall, setIsInCall] = useState(true);

  const SOCKET_URL = "wss://audio.enty.services/stream-vc";
  let wavStreamPlayer: WavStreamPlayer | null = null;

  // Initialize WavStreamPlayer
  useEffect(() => {
    wavStreamPlayer = new WavStreamPlayer({ sampleRate: 24000 });

    // Connect to audio output
    wavStreamPlayer.connect().catch((error) => {
      console.error("Error connecting to audio output:", error);
    });

    return () => {
      if (wavStreamPlayer) {
      }
    };
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
                            prosody: { speed: 1.0, volume: 0 },
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

              try {
                let audioData: ArrayBuffer;

                if (event.data instanceof ArrayBuffer) {
                  audioData = event.data;
                  bufferAudio(audioData);
                } else if (event.data instanceof Blob) {
                  const reader = new FileReader();
                  reader.onloadend = () => {
                    audioData = reader.result as ArrayBuffer;
                    bufferAudio(audioData);
                  };
                  reader.readAsArrayBuffer(event.data);
                  return;
                } else {
                  throw new Error("Received unexpected data type from WebSocket");
                }

                bufferAudio(audioData);
              } catch (error) {
                console.error("Error processing WebSocket message:", error);
              }
            };

            websocket.onclose = () => {
              console.log("WebSocket connection closed...");
              if (!isInCall) {
                setConnectionStatus("Reconnecting...");
                setTimeout(reconnectWebSocket, 5000);
              }
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

  // Buffer audio and add it to the queue using WavStreamPlayer
  function bufferAudio(data: ArrayBuffer) {
    if (wavStreamPlayer) {
      const audio16BitPCM = new Int16Array(data);
      wavStreamPlayer.add16BitPCM(audio16BitPCM, 'my-track'); // Add PCM data to the player

      // You can also track frequency data if needed
      const frequencyData = wavStreamPlayer.getFrequencies();
      console.log(frequencyData);

      // If playback is interrupted, restart it by adding more PCM data
      // wavStreamPlayer.interrupt().then((trackOffset) => {
      //   console.log(trackOffset);
      // });
    }
  }

  function arrayBufferToBase64(arrayBuffer: ArrayBuffer): string {
    let binary = '';
    const uint8Array = new Uint8Array(arrayBuffer);
    const len = uint8Array.length;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(uint8Array[i]);
    }
    return btoa(binary);
  }

  async function infiniteLoop() {
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 1000)); // 每秒钟迭代一次
    }
  }

  // End call function
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

    await infiniteLoop();
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
            <span className={isInCall ? isPlayingAudio ? styles.speakingAnimation : styles.listeningAnimation : styles.offlineAnimation}>
              {isInCall ? isPlayingAudio ? "AI正在说话" : "AI正在听" : "AI 离线"}
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
