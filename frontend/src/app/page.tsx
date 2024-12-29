"use client";

import { useEffect, useState, useRef } from "react";
import styles from "./page.module.css";
import msgpack from 'msgpack-lite';
import { useMicVAD, utils } from "@ricky0123/vad-react";

export default function Home() {
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(true); // true means listening, false means speaking
  const [isPlayingAudio, setIsPlayingAudio] = useState(false); // State to track audio playback
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
  const [audioDuration, setAudioDuration] = useState<number>(0); // State to track audio duration

  const [isCallEnded, setIsCallEnded] = useState(false); // Add this state

  // VAD - Voice Activity Detection
  const [audioList, setAudioList] = useState<string[]>([]);
  const vad = useMicVAD({
    model: "v5",
    baseAssetPath: "/",
    onnxWASMBasePath: "/",
    onSpeechEnd: (audio: blob) => {
      const wavBuffer = utils.encodeWAV(audio);
      const base64 = utils.arrayBufferToBase64(wavBuffer);
      const url = `data:audio/wav;base64,${base64}`;
      setAudioList((old) => [url, ...old]);
    },
  });

  // Connection status and other states
  type ConnectionStatus = "Connecting..." | "Connected" | "Disconnected" | "Closed";
  const [connectionStatus, setConnectionStatus] = useState<string>("Connecting...");
  const [isInCall, setIsInCall] = useState(true);

  let manualClose = false;
  let audioContext: AudioContext | null = null;
  let audioBufferQueue: AudioBuffer[] = [];

  if (typeof window !== "undefined" && window.AudioContext) {
    audioContext = new AudioContext();
  }

  const audioManager = {
    stopCurrentAudio: () => {
      if (isPlayingAudio) {
        setIsPlayingAudio(false);
      }
    },
    playNewAudio: async (audioBlob: Blob) => {
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      audio.onloadedmetadata = () => {
        setAudioDuration(audio.duration);
      };

      setIsPlayingAudio(true);

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        setIsPlayingAudio(false);
        setIsRecording(true);

        if (audioQueue.length > 0) {
          const nextAudioBlob = audioQueue.shift();
          if (nextAudioBlob) {
            audioManager.playNewAudio(nextAudioBlob);
          }
        }
      };

      try {
        await audio.play();
      } catch (error) {
        console.error("播放音频失败:", error);
        audioManager.stopCurrentAudio();
      }
    }
  };

  // Handle WebSocket connection
  const SOCKET_URL = "wss://audio.enty.services/stream";
  useEffect(() => {
    let wakeLock: WakeLockSentinel | null = null;

    async function requestWakeLock() {
      try {
        wakeLock = await navigator.wakeLock.request("screen");
        console.log("Screen wake lock acquired");

        const script = document.createElement("script");
        script.src = "https://www.WebRTC-Experiment.com/RecordRTC.js";
        script.onload = () => {
          const RecordRTC = (window as any).RecordRTC;
          const StereoAudioRecorder = (window as any).StereoAudioRecorder;

          if (navigator) {
            navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
              console.log("RecordRTC start.");
              const reconnectWebSocket = () => {
                if (manualClose || isCallEnded) {
                  console.log("Reconnection prevented by manualClose or isCallEnded flag.");
                  return;
                }

                const websocket = new WebSocket(SOCKET_URL);
                setSocket(websocket);

                websocket.onopen = () => {
                  console.log("client connected to websocket");
                  setConnectionStatus("Connected");
                  setIsInCall(true);
                  const recorder = new RecordRTC(stream, {
                    type: 'audio',
                    recorderType: StereoAudioRecorder,
                    mimeType: 'audio/wav',
                    timeSlice: 100,
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
                                vc_uid: "c9cf4e49"
                              }
                            };
                            const encodedData = JSON.stringify(message);
                            if (websocket) {
                              websocket.send(encodedData);
                            }
                          }
                        };
                        reader.readAsArrayBuffer(blob);
                      }
                    }
                  });

                  recorder.startRecording();
                };

                websocket.onmessage = (event) => {
                  try {
                   if (event.data instanceof Blob) {
                      const reader = new FileReader();
                      reader.onloadend = () => {
                        checkAndBufferAudio(reader.result as ArrayBuffer);
                      };
                      reader.readAsArrayBuffer(event.data);
                      return;
                    }
                  } catch (error) {
                    console.error("Error processing WebSocket message:", error);
                  }
                };

                websocket.onclose = () => {
                  if (manualClose || isCallEnded) return;
                  if (connectionStatus === "Closed") {
                    console.log("WebSocket 已关闭");
                    return;
                  }
                  console.log("WebSocket connection closed...");
                  setConnectionStatus("Reconnecting...");
                  setTimeout(reconnectWebSocket, 5000);
                };

                websocket.onerror = (error) => {
                  console.error("WebSocket error:", error);
                  websocket?.close();
                };
              };

              if (manualClose || isCallEnded) return;
              reconnectWebSocket();
            }).catch((error) => {
              console.error("Error with getUserMedia", error);
            });
          }
        };
        document.body.appendChild(script);
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

  function checkAndBufferAudio(audioData: ArrayBuffer) {
    const text = new TextDecoder("utf-8").decode(audioData);
    if (text.includes("END_OF_AUDIO")) {
      console.log("Detected END_OF_AUDIO signal in audioData");
      audioManager.stopCurrentAudio();
      setIsRecording(true);
      setIsPlayingAudio(false);
      return;
    }
    bufferAudio(audioData);
  }

  function bufferAudio(data: ArrayBuffer) {
    if (audioContext) {
      audioContext.decodeAudioData(data, (buffer) => {
        audioBufferQueue.push(buffer);

        if (!isPlayingAudio) {
          playAudioBufferQueue();
        }
      });
    }
  }

  function playAudioBufferQueue() {
    if (audioBufferQueue.length === 0) {
      setIsPlayingAudio(false);
      setIsRecording(true);
      return;
    }

    const buffer = audioBufferQueue.shift();
    if (buffer && audioContext) {
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.destination);

      source.onended = () => {
        playAudioBufferQueue();
      };

      source.start();
      setIsPlayingAudio(true);
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

  // End Call Handler
  const endCall = async () => {
    manualClose = true;
    setConnectionStatus("Closed");
    setIsCallEnded(true);

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
  };

  return (
    <div className={styles.container}>
      <div className={styles.statusBar}>
        <div className={styles.connectionStatus}>
          <div className={`${styles.statusDot} ${connectionStatus === "Connected" ? styles.connected : ""}`} />
          {connectionStatus}
        </div>
      </div>

      <div className={styles.mainContent}>
        <div className={styles.avatarSection}>
          <div className={`${styles.avatarContainer} ${isPlayingAudio ? styles.speaking : ""}`}>
            <img src="/ai-avatar.png" alt="AI" className={styles.avatar} />
          </div>
          <div className={styles.status}>
            <span className={isInCall ? (isPlayingAudio ? styles.speakingAnimation : styles.listeningAnimation) : styles.offlineAnimation}>
              {isInCall ? (isPlayingAudio ? "AI正在说话" : "AI正在听") : "AI 离线"}
            </span>
          </div>
        </div>
      </div>

      {/* Add the VAD status */}
      <div>
        <h6>Listening</h6>
        {!vad.listening && "Not"} listening
        <h6>Loading</h6>
        {!vad.loading && "Not"} loading
        <h6>Errored</h6>
        {!vad.errored && "Not"} errored
        <h6>User Speaking</h6>
        {!vad.userSpeaking && "Not"} speaking
        <h6>Audio count</h6>
        {audioList.length}
        <h6>Start/Pause</h6>
        <button onClick={vad.pause}>Pause</button>
        <button onClick={vad.start}>Start</button>
        <button onClick={vad.toggle}>Toggle</button>
      </div>

      <div className={styles.controls}>
        <button className={isInCall ? styles.endCallButton : styles.startCallButton} onClick={() => {
          if (isInCall) {
            endCall();
          } else {
            window.location.reload();
          }
        }}>
          {isInCall ? "结束通话" : "重新通话"}
        </button>
      </div>
    </div>
  );
}
