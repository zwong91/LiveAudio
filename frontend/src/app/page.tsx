"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.css";

const [isRecording, setIsRecording] = useState(true); // true means listening, false means speaking
const [isPlayingAudio, setIsPlayingAudio] = useState(false); // State to track audio playback
const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
const [audioDuration, setAudioDuration] = useState<number>(0); // State to track audio duration

// 音频管理器
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


// WebRTC 管理器
const useWebRTC = (
  BASE_URL: string,
  audioQueue: Blob[],
  setAudioQueue: Function,
  setIsRecording: Function,
  checkAndBufferAudio: Function
) => {
  const [connectionStatus, setConnectionStatus] = useState("Connecting...");
  const [isCallEnded, setIsCallEnded] = useState(false);
  const [peerConnection, setPeerConnection] = useState<RTCPeerConnection | null>(null);
  const [dataChannel, setDataChannel] = useState<RTCDataChannel | null>(null);
  useEffect(() => {
    // Ensure WebRTC only runs in the browser
    if (typeof window !== "undefined" && window.RTCPeerConnection) {
      const pc = new RTCPeerConnection();
      setPeerConnection(pc);

      const setupConnection = async () => {
        try {
          // 获取音频流
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          // 添加音频轨道到 PeerConnection
          stream.getTracks().forEach((track) => {
            console.log("Adding track to connection:", track);
            pc.addTransceiver(track, { direction: "sendrecv" });
          });

          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);

          // 将 SDP 和 type 数据包装成一个 JSON 对象
          const offerData = {
            sdp: offer.sdp,
            type: offer.type  // 添加 type 字段
          };
          const response = await fetch('https://gtp.aleopool.cc/offer', {
            method: "POST",
            body: JSON.stringify(offerData),  // 转换为 JSON 字符串
            headers: { "Content-Type": "application/json" },
          });

          const answer = await response.json();
          await pc.setRemoteDescription({ sdp: answer.sdp, type: "answer" });
          setConnectionStatus("Connected");
        } catch (error) {
          console.error("WebRTC 初始化失败:", error);
          setConnectionStatus("Disconnected");
        }
      };

      setupConnection();

      // 创建 DataChannel 对象
      const dc = pc.createDataChannel('response');
      setDataChannel(dc);
  
      return () => {
        pc.close();
        setConnectionStatus("Closed");
      };
    } else {
      setConnectionStatus("WebRTC not supported");
    }
  }, [BASE_URL]);

  useEffect(() => {
    if (peerConnection) {
      // // Handle inbound tracks
      // peerConnection.ontrack = (event: RTCTrackEvent) => {
      //   console.log("Inbound track:", event.track.kind);

      //   // Create an <audio> element for audio tracks
      //   if (event.track.kind === "audio") {
      //     const el = document.createElement('audio');
      //     el.srcObject = event.streams[0];
      //     el.autoplay = el.controls = true;
      //     el.style.maxWidth = "100%";
      //     document.body.appendChild(el); // Append to the body or any other container you prefer
      //     console.log("Audio track added to page");
      //   }
      // };
      peerConnection.ondatachannel = (event: RTCDataChannelEvent) => {
        const dataChannel = event.channel;
  
        dataChannel.onopen = () => {
          console.log("DataChannel opened:", dataChannel.label);
          dataChannel.send("hah")
        };
  
        dataChannel.onmessage = async (event: MessageEvent) => {
          console.log("Received message:", event.data);
          try {
            let audioData: ArrayBuffer;
  
            if (event.data instanceof ArrayBuffer) {
              audioData = event.data;
            } else if (event.data instanceof Blob) {
              const arrayBuffer = await event.data.arrayBuffer();
              audioData = arrayBuffer;
            } else {
              throw new Error("Unsupported data type received");
            }
  
            checkAndBufferAudio(audioData);
          } catch (error) {
            console.error("Error processing WebSocket message:", error);
          }
          dataChannel.send("hah")
        };
  
        dataChannel.onclose = () => {
          console.log("DataChannel closed:", dataChannel.label);
        };
      };
    }
  }, [peerConnection, checkAndBufferAudio]);  

  return {
    connectionStatus,
    isCallEnded,
    endCall: () => {
      if (peerConnection) {
        peerConnection.close();
      }
      setConnectionStatus("Closed");
      setIsCallEnded(true);
    },
  };
};


// 主组件
export default function Home() {
  //const BASE_URL = "https://audio.enty.services";
  const BASE_URL = "https://gtp.aleopool.cc";
  const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
  const [isRecording, setIsRecording] = useState(true);

  const { isPlayingAudio, playAudio, checkAndBufferAudio, stopCurrentAudio } = useAudioManager(
    audioQueue,
    setAudioQueue,
    setIsRecording
  );
  const { connectionStatus, isCallEnded, endCall } = useWebRTC(
    BASE_URL,
    audioQueue,
    setAudioQueue,
    setIsRecording,
    checkAndBufferAudio
  );

  useEffect(() => {
    if (!isPlayingAudio && audioQueue.length > 0) {
      const nextAudioBlob = audioQueue.shift();
      if (nextAudioBlob) playAudio(nextAudioBlob);
    }
  }, [isPlayingAudio, audioQueue, playAudio]);

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
                connectionStatus === "Closed"
                  ? styles.offlineAnimation
                  : isPlayingAudio
                  ? styles.speakingAnimation
                  : styles.listeningAnimation
              }
            >
              {connectionStatus === "Closed"
                ? "AI 离线"
                : isPlayingAudio
                ? "AI正在说话"
                : "AI正在听"}
            </span>
          </div>
        </div>
      </div>

      <div className={styles.controls}>
        <button
          className={isCallEnded ? styles.startCallButton : styles.endCallButton}
          onClick={endCall}
        >
          {isCallEnded ? "重新通话" : "结束通话"}
        </button>
      </div>
    </div>
  );
}
