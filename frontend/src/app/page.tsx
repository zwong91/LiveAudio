"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.css";

// 音频管理器
const useAudioManager = (audioQueue: Blob[], setAudioQueue: Function, setIsRecording: Function) => {
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioDuration, setAudioDuration] = useState<number>(0);
  const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null); // 追踪当前播放的音频

  const stopCurrentAudio = () => {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.currentTime = 0;
      setIsPlayingAudio(false);
    }
  };

  const playAudio = async (audioBlob: Blob) => {
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    setCurrentAudio(audio); // 设置当前播放的音频对象

    audio.onloadedmetadata = () => setAudioDuration(audio.duration);

    audio.onended = () => {
      URL.revokeObjectURL(audioUrl);
      setIsPlayingAudio(false);
      setIsRecording(true);

      if (audioQueue.length > 0) {
        const nextAudioBlob = audioQueue.shift();
        if (nextAudioBlob) playAudio(nextAudioBlob);
      }
    };

    try {
      setIsPlayingAudio(true);
      await audio.play();
    } catch (error) {
      console.error("播放音频失败:", error);
      setIsPlayingAudio(false);
    }
  };

  const checkAndBufferAudio = (audioData: ArrayBuffer) => {
    const text = new TextDecoder("utf-8").decode(audioData);

    if (text.includes("END_OF_AUDIO")) {
      console.log("Detected END_OF_AUDIO signal in audioData");
      stopCurrentAudio(); // 停止当前音频播放
      setIsRecording(true);
      setIsPlayingAudio(false);
      return;
    }

    // 如果没有检测到 "END_OF_AUDIO" 信号，继续缓存音频并立即播放
    const audioBlob = new Blob([audioData], { type: "audio/wav" });
    setAudioQueue((prevQueue: Blob[]) => {
      const newQueue = [...prevQueue, audioBlob];
      // 播放新的音频
      if (!isPlayingAudio) {
        playAudio(audioBlob); // 立刻播放当前音频
      }
      return newQueue;
    });
  };

  return {
    isPlayingAudio,
    audioDuration,
    playAudio,
    checkAndBufferAudio,
    stopCurrentAudio,
  };
};

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

  useEffect(() => {
    // Ensure WebRTC only runs in the browser
    if (typeof window !== "undefined" && window.RTCPeerConnection) {
      const pc = new RTCPeerConnection();
      setPeerConnection(pc);

      const setupConnection = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          stream.getTracks().forEach((track) =>
            pc.addTransceiver(track, { direction: "sendrecv" })
          );

          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);

          const response = await fetch(`${BASE_URL}/offer`, {
            method: "POST",
            body: offer.sdp,
            headers: { "Content-Type": "application/sdp" },
          });

          const answer = await response.text();
          await pc.setRemoteDescription({ sdp: answer, type: "answer" });

          setConnectionStatus("Connected");
        } catch (error) {
          console.error("WebRTC 初始化失败:", error);
          setConnectionStatus("Disconnected");
        }
      };

      setupConnection();

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
      peerConnection.ondatachannel = (event: RTCDataChannelEvent) => {
        const dataChannel = event.channel;
  
        dataChannel.onopen = () => {
          console.log("DataChannel opened:", dataChannel.label);
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
  const BASE_URL = "wss://gtp.aleopool.cc/stream";
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
