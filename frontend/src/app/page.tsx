"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.css";

import { useState } from 'react';

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

      if (audioQueue.length > 0) {
        const nextAudioBlob = audioQueue.shift();
        if (nextAudioBlob) {
          playAudio(nextAudioBlob);
        }
      } else {
        // 播放完所有音频后清空队列
        setAudioQueue([]);
        setIsRecording(true);
      }
    };

    try {
      setIsPlayingAudio(true);
      await audioplay();
    } catch (error) {
      console.error("播放音频失败:", error);
      setIsPlayingAudio(false);
    }
  };

  const checkAndBufferAudio = (audioData: ArrayBuffer) => {
    const text = new TextDecoder("utf-8").decode(audioData);

    if (text.includes("END_OF_AUDIO")) {
      console.log("Detected ENDOfAudio signal in audioData");
      stopCurrentAudio(); // 停止当前音频播放
      setIsRecording(true);
      return;
    }

    // 如果没有检测到 "END_OF_AUDIO" 信号，继续缓存音频并播放
    const audioBlob = new Blob([audioData], { type: "audio/wav"});
    setAudioQueue((prevQueue: Blob[]) => {
      const newQueue = [...prevQueue, audioBlob];
      playNextAudio();
      return newQueue;
    });
  };

  const playNextAudio = () => {
    if (!isPlayingAudio && audioQueue.length > 0) {
      const nextAudioBlob = audioQueue.shift();
      if (nextAudioBlob) {
        playAudio(nextAudioBlob);
      }
    }
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
              audioData = await event.data.arrayBuffer();
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
