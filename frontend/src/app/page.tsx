"use client";

import { useEffect, useState, useRef } from "react";
import styles from "./page.module.css";
import msgpack from 'msgpack-lite';

export default function Home() {
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(true); // true means listening, false means speaking
  const [isPlayingAudio, setIsPlayingAudio] = useState(false); // State to track audio playback
  const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
  const [audioDuration, setAudioDuration] = useState<number>(0); // State to track audio duration

  const [isCallEnded, setIsCallEnded] = useState(false); // Add this state

  // 定义可能的连接状态类型
  type ConnectionStatus = "Connecting..." | "Connected" | "Disconnected" | "Closed";

  const [connectionStatus, setConnectionStatus] = useState<string>("Connecting..."); // State to track connection status

  let manualClose = false;
  let audioContext: AudioContext | null = null;
  let audioBufferQueue: AudioBuffer[] = [];

  // Check if AudioContext is available in the browser
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

      // When the metadata of the audio is loaded, set its duration
      audio.onloadedmetadata = () => {
        setAudioDuration(audio.duration); // Set the audio duration after loading metadata
      };

      // Play the audio
      setIsPlayingAudio(true);

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        setIsPlayingAudio(false);
        setIsRecording(true);

        if (audioQueue.length > 0) {
          const nextAudioBlob = audioQueue.shift();
          if (nextAudioBlob) {
            audioManager.playNewAudio(nextAudioBlob); // Play next audio in the queue
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

  // 检查 ArrayBuffer 是否包含 "END_OF_AUDIO" 并处理音频数据
  function checkAndBufferAudio(audioData: ArrayBuffer) {
    // 将 ArrayBuffer 转为字符串
    const text = new TextDecoder("utf-8").decode(audioData);

    if (text.includes("END_OF_AUDIO")) {
      console.log("Detected END_OF_AUDIO signal in audioData");
      // 停止当前音频播放
      audioManager.stopCurrentAudio();
      // 停止录音并切换状态
      setIsRecording(true);
      setIsPlayingAudio(false);
      return;
    }
    // 如果不包含 END_OF_AUDIO，则缓冲音频数据
    bufferAudio(audioData);
  }

  // Buffer audio and add it to the queue
  function bufferAudio(data: ArrayBuffer) {
    if (audioContext) {
      audioContext.decodeAudioData(data, (buffer) => {
        // Buffer the audio chunk and push it to the queue
        audioBufferQueue.push(buffer);

        // If we are not already playing, start playing the audio
        if (!isPlayingAudio) {
          playAudioBufferQueue();
        }
      });
    }
  }

  // Play the buffered audio chunks from the queue
  function playAudioBufferQueue() {
    if (audioBufferQueue.length === 0) {
      setIsPlayingAudio(false); // Stop playback if queue is empty
      setIsRecording(true); // Start recording again
      return;
    }

    const buffer = audioBufferQueue.shift(); // Get the next audio buffer
    if (buffer && audioContext) {
      const source = audioContext.createBufferSource();
      source.buffer = buffer;

      // Connect the source to the audio context's output
      source.connect(audioContext.destination);

      // When this audio ends, play the next one
      source.onended = () => {
        playAudioBufferQueue(); // Continue playing the next buffer
      };

      // Start playing the audio
      source.start();

      // Update the state to reflect the playing status
      setIsPlayingAudio(true);
    }
  }

  const fns = {
    // Function to get the HTML of the page
    getPageHTML: () => {
      return { success: true, html: document.documentElement.outerHTML };
    },
    
    // Function to change the background color
    changeBackgroundColor: ({ color }: { color: string }) => {
      document.body.style.backgroundColor = color;
      return { success: true, color };
    },
    
    // Function to change the text color
    changeTextColor: ({ color }: { color: string }) => {
      document.body.style.color = color;
      return { success: true, color };
    },
  };
  
  
  const BASE_URL = "https://audio.enty.services";

  // ICE server configuration (STUN/TURN servers)
  const iceServers = [
    { urls: 'stun:stun.l.google.com:19302' }, // STUN server
    // { 
    //   urls: 'turn:your.turnserver.com', // TURN server
    //   username: 'your_username',
    //   credential: 'your_credential'
    // }
  ];

  // Create a new RTCPeerConnection WebRTC Agent with ICE servers
  const peerConnection = new RTCPeerConnection({ iceServers });
  
  // On inbound audio add to page
  peerConnection.ontrack = (event) => {
    const el = document.createElement('audio');
    el.srcObject = event.streams[0];
    el.autoplay = el.controls = true;
    document.body.appendChild(el);
  };
  
  const dataChannel = peerConnection.createDataChannel('response');
  
  function configureData() {
    console.log('Configuring data channel');
    const event = {
      type: 'session.update',
      session: {
        modalities: ['text', 'audio'],
        // Provide the tools. Note they match the keys in the `fns` object above
        tools: [
          {
            type: 'function',
            name: 'changeBackgroundColor',
            description: 'Changes the background color of a web page',
            parameters: {
              type: 'object',
              properties: {
                color: { type: 'string', description: 'A hex value of the color' },
              },
            },
          },
          {
            type: 'function',
            name: 'changeTextColor',
            description: 'Changes the text color of a web page',
            parameters: {
              type: 'object',
              properties: {
                color: { type: 'string', description: 'A hex value of the color' },
              },
            },
          },
          {
            type: 'function',
            name: 'getPageHTML',
            description: 'Gets the HTML for the current page',
          },
        ],
      },
    };
    dataChannel.send(JSON.stringify(event));
  }
  
  dataChannel.addEventListener('open', (ev) => {
    console.log('Opening data channel', ev);
    configureData();
  });
  
  // {
  //     "type": "response.function_call_arguments.done",
  //     "event_id": "event_Ad2gt864G595umbCs2aF9",
  //     "response_id": "resp_Ad2griUWUjsyeLyAVtTtt",
  //     "item_id": "item_Ad2gsxA84w9GgEvFwW1Ex",
  //     "output_index": 1,
  //     "call_id": "call_PG12S5ER7l7HrvZz",
  //     "name": "get_weather",
  //     "arguments": "{\"location\":\"Portland, Oregon\"}"
  // }
  interface FunctionResponse {
    type: string;
    event_id: string;
    response_id: string;
    item_id: string;
    output_index: number;
    call_id: string;
    name: keyof typeof fns;  // `name` is one of the keys in `fns`
    arguments: string;
  }

  dataChannel.addEventListener('message', async (ev) => {
    const msg: FunctionResponse = JSON.parse(ev.data);
  
    // Handle function calls
    if (msg.type === 'response.function_call_arguments.done') {
      const fn = fns[msg.name];
      if (fn !== undefined) {
        console.log(`Calling local function ${msg.name} with ${msg.arguments}`);
        const args = JSON.parse(msg.arguments);
        const result = await fn(args);
        console.log('result', result);
  
        // Let OpenAI know that the function has been called and share its output
        const event = {
          type: 'conversation.item.create',
          item: {
            type: 'function_call_output',
            call_id: msg.call_id, // call_id from the function_call message
            output: JSON.stringify(result), // result of the function
          },
        };
        dataChannel.send(JSON.stringify(event));
      }
    }
  });

  // Initialize WebSocket and media devices
  useEffect(() => {
    let wakeLock: WakeLockSentinel | null = null;

    // Request screen wake lock to prevent the screen from going to sleep
    async function requestWakeLock() {
      try {
        wakeLock = await navigator.wakeLock.request("screen");
        console.log("Screen wake lock acquired");
        // Capture microphone
        navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
          // Add microphone to PeerConnection
          stream.getTracks().forEach((track) => peerConnection.addTransceiver(track, { direction: 'sendrecv' }));

          peerConnection.createOffer().then((offer) => {
            peerConnection.setLocalDescription(offer);

            // Send WebRTC Offer to Workers Realtime WebRTC API Relay
            fetch(`${BASE_URL}/offer`, {
              method: 'POST',
              body: offer.sdp,
              headers: {
                'Content-Type': 'application/sdp',
              },
            })
              .then((r) => r.text())
              .then((answer) => {
                // Accept answer from Realtime WebRTC API
                peerConnection.setRemoteDescription({
                  sdp: answer,
                  type: 'answer',
                });
              });
          });
        });

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
  }, [manualClose, isCallEnded]);

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

  // 添加状态来跟踪是否在通话中
  const [isInCall, setIsInCall] = useState(true);

  const endCall = async () => {
    manualClose = true;
    setConnectionStatus("Closed");
    setIsCallEnded(true); // Set isCallEnded to true to prevent reconnection

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
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
            <span className={isInCall ? (isPlayingAudio ? styles.speakingAnimation : styles.listeningAnimation) : styles.offlineAnimation}>
              {isInCall ? (isPlayingAudio ? "AI正在说话" : "AI正在听") : "AI 离线"}
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