"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.css";

// 音频管理器
const useAudioManager = (audioQueue: Blob[], setAudioQueue: Function, setIsRecording: Function) => {
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioDuration, setAudioDuration] = useState<number>(0);

  const playAudio = async (audioBlob: Blob) => {
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);

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

  return {
    isPlayingAudio,
    audioDuration,
    playAudio,
  };
};

// WebRTC 管理器
const useWebRTC = (BASE_URL: string, audioQueue: Blob[], setAudioQueue: Function, setIsRecording: Function) => {
  const [connectionStatus, setConnectionStatus] = useState("Connecting...");
  const [isCallEnded, setIsCallEnded] = useState(false);
  const peerConnection = new RTCPeerConnection();

  useEffect(() => {
    const setupConnection = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((track) =>
          peerConnection.addTransceiver(track, { direction: "sendrecv" })
        );

        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);

        const response = await fetch(`${BASE_URL}/offer`, {
          method: "POST",
          body: offer.sdp,
          headers: { "Content-Type": "application/sdp" },
        });

        const answer = await response.text();
        await peerConnection.setRemoteDescription({ sdp: answer, type: "answer" });

        setConnectionStatus("Connected");
      } catch (error) {
        console.error("WebRTC 初始化失败:", error);
        setConnectionStatus("Disconnected");
      }
    };

    setupConnection();

    return () => {
      peerConnection.close();
      setConnectionStatus("Closed");
    };
  }, [BASE_URL, peerConnection]);

  // WebRTC 音频数据处理
  useEffect(() => {
    peerConnection.ontrack = (event) => {
      setIsRecording(false);
      const audioBlob = new Blob([event.streams[0]], { type: "audio/wav" });
      setAudioQueue((prevQueue: Blob[]) => [...prevQueue, audioBlob]);
    };
  }, [peerConnection, setAudioQueue, setIsRecording]);

  return {
    connectionStatus,
    isCallEnded,
    endCall: () => {
      peerConnection.close();
      setConnectionStatus("Closed");
      setIsCallEnded(true);
    },
  };
};

// 主组件
export default function Home() {
  const BASE_URL = "https://audio.enty.services";
  const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
  const [isRecording, setIsRecording] = useState(true);

  const { isPlayingAudio, playAudio } = useAudioManager(audioQueue, setAudioQueue, setIsRecording);
  const { connectionStatus, isCallEnded, endCall } = useWebRTC(BASE_URL, audioQueue, setAudioQueue, setIsRecording);

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
