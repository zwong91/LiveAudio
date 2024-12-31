"use client";

import { useEffect, useState, useRef } from "react";
import styles from "./page.module.css";
import { useMicVAD, utils } from "@ricky0123/vad-react"

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
  audioQueue: Blob[],
  setAudioQueue: Function,
  setIsRecording: Function,
  checkAndBufferAudio: Function
) => {
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [isCallEnded, setIsCallEnded] = useState(false);
  const [peerConnection, setPeerConnection] = useState<RTCPeerConnection | null>(null);
  const [dataChannel, setDataChannel] = useState<RTCDataChannel | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [reconnectTimer, setReconnectTimer] = useState<NodeJS.Timeout | null>(null);
  useEffect(() => {
    // Ensure WebRTC only runs in the browser
    if (typeof window !== "undefined" && window.RTCPeerConnection) {

      // ICE 服务器配置
      const iceServers = [
        // {
        //   urls: [
        //     "stun:stun.l.google.com:19302",    // Google STUN 服务器
        //     "stun:stun1.l.google.com:19302",   // 备用 Google STUN 服务器
        //     "stun:stun2.l.google.com:19302",   // 备用 Google STUN 服务器
        //     "stun:stun3.l.google.com:19302",   // 备用 Google STUN 服务器
        //     "stun:stun4.l.google.com:19302"    // 备用 Google STUN 服务器
        //   ]
        // },
        { urls: 'stun:audio.enty.services:3478' },
        //如果需要，可以添加 TURN 服务器
        {
          urls: "turn:audio.enty.services:3478",   // TURN 服务器
          username: "admin",            // TURN 服务器用户名
          credential: "7f0dd067662502af36934e85b43895b148edfcdb", // TURN 服务器密码
          credentialType: 'password',
          realm: 'audio.enty.services',
        },
      ];
      

      // 配置 ICE 服务器
      const pcConfig = {
        iceServers: iceServers,
      };
      
      const pc = new RTCPeerConnection(pcConfig);
      //const pc = new RTCPeerConnection();
      setPeerConnection(pc);
      const setupConnection = async () => {
        try {
          pc.oniceconnectionstatechange = (event) => {
              const state = pc.iceConnectionState;
              console.log("ICE connection state:", state);
              if (state === "disconnected" || state === "failed") {
                setConnectionStatus("disconnected");
                if (reconnectAttempts < 5) { // 限制重试次数
                  setReconnectAttempts(reconnectAttempts + 1);
                  setReconnectTimer(setTimeout(() => {
                    pc.close();
                    setConnectionStatus("reconnecting");
                    setupConnection();
                  }, 5000)); // 5秒后重试
                }
              }
          };
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
          const response = await fetch('api/rtc-connect', {
            method: "POST",
            body: JSON.stringify(offerData),  // 转换为 JSON 字符串
            headers: { "Content-Type": "application/json" },
          });

          const answer = await response.json();
          await pc.setRemoteDescription({ sdp: answer.sdp, type: "answer" });
        } catch (error) {
          console.error("WebRTC 初始化失败:", error);
          setConnectionStatus("disconnected");
        }
      };

      setupConnection();

      // 创建 DataChannel 对象, 触发ICE协商
      const dc = pc.createDataChannel('response');
      setDataChannel(dc);

      return () => {
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
        }
        if (peerConnection) {
            peerConnection.close();
            setConnectionStatus("disconnected");
        }
      };
    } else {
      setConnectionStatus("WebRTC not supported");
    }
  }, [reconnectAttempts]);

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

      // const sendIceCandidateToServer = async (candidate: RTCIceCandidate) => {
      //   try {
      //     // 创建 candidate 数据
      //     const candidateData = {
      //       candidate: candidate.candidate,
      //       sdpMid: candidate.sdpMid,
      //       sdpMLineIndex: candidate.sdpMLineIndex,
      //     };
      
      //     // 将 candidate 发送到服务器
      //     const response = await fetch('/api/rtc-ice-candidate', {
      //       method: 'POST',
      //       body: JSON.stringify(candidateData),
      //       headers: { 'Content-Type': 'application/json' },
      //     });
      
      //     if (!response.ok) {
      //       console.error('Failed to send ICE candidate');
      //     }
      //   } catch (error) {
      //     console.error('Error sending ICE candidate:', error);
      //   }
      // };
      
      peerConnection.onicecandidate = (event: RTCPeerConnectionIceEvent) => {
        if (event.candidate) {
          console.log('获取到ICE候选:', event.candidate.type, event.candidate.address);
          if (event.candidate.type === 'srflx') {
            console.log('STUN成功！');
            console.log('公网IP:', event.candidate.address);
            console.log('公网端口:', event.candidate.port);
          }
          if (event.candidate.type === 'relay') {
            console.log('TURN成功！');
            console.log('中继IP:', event.candidate.address);
            console.log('中继端口:', event.candidate.port);
          }
          // 发送 ICE candidate 到远端
          //sendIceCandidateToServer(event.candidate);
        }
      };
      peerConnection.onconnectionstatechange = (event: Event) => {
        let state = (event.target as RTCPeerConnection).connectionState;
        console.log("on connectionstate changed:", state);
        if (state == 'failed')
          state = "disconnected"
        setConnectionStatus(state);
      };
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
      setConnectionStatus("disconnected");
      setIsCallEnded(true);
    },
  };
};


// 主组件
export default function Home() {
  const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
  const [isRecording, setIsRecording] = useState(true);
  const [audioList, setAudioList] = useState<string[]>([]);

  const audioItemKey = (audioURL: string) => audioURL.substring(-10)
  const vad = useMicVAD({
    model: "v5",
    baseAssetPath: "/",
    onnxWASMBasePath: "/",
    onSpeechEnd: (audio: Float32Array) => {
      const wavBuffer = utils.encodeWAV(audio);
      const base64 = utils.arrayBufferToBase64(wavBuffer);
      const url = `data:audio/wav;base64,${base64}`;
      setAudioList((old) => [url, ...old]);
    },
  });


  const { isPlayingAudio, playAudio, checkAndBufferAudio, stopCurrentAudio } = useAudioManager(
    audioQueue,
    setAudioQueue,
    setIsRecording
  );
  const { connectionStatus, isCallEnded, endCall } = useWebRTC(
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

  // Integrate Eruda
  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/eruda';
    script.onload = () => {
      (window as any).eruda.init();
    };
    document.body.appendChild(script);
  }, []);

  return (
    <div className={styles.container}>
      <div className={styles.statusBar}>
        <div className={styles.connectionStatus}>
          <div
            className={`${styles.statusDot} ${
              connectionStatus === "connected" ? styles.connected : ""
            }`}
          />
          {connectionStatus}
        </div>
      </div>

      <div className={styles.mainContent}>
        <div className={styles.avatarSection}>
          <div className={`${styles.avatarContainer} ${isPlayingAudio ? styles.speaking : ""}`}>
            <img src="/ai-avatar.png" alt="AI" className={styles.avatar} />
          </div>
          <div className={styles.status}>
            <span
              className={
                connectionStatus === "disconnected"
                  ? styles.offlineAnimation
                  : isPlayingAudio
                  ? styles.speakingAnimation
                  : styles.listeningAnimation
              }
            >
              {connectionStatus === "disconnected"
                ? "AI 离线"
                : isPlayingAudio
                ? "AI正在说话"
                : "AI正在听"}

            </span>
          </div>
        </div>
      </div>

      <div>
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

      {/* Add the audio playlist */}
      <div>
        <ol
          id="playlist"
          className="self-center pl-0 max-h-[400px] overflow-y-auto no-scrollbar list-none"
        >
          {audioList.map((audioURL) => {
            return (
              <li className="pl-0" key={audioItemKey(audioURL)}>
                <audio src={audioURL} controls />
              </li>
            );
          })}
        </ol>
      </div>
    </div>

      <div className={styles.controls}>
        <button
          className={!isCallEnded ? styles.startCallButton : styles.endCallButton}
          onClick={() => {
            if (!isCallEnded) {
              endCall();
            } else {
              window.location.reload();
            }
          }}
        >
          {isCallEnded ? "重新通话" : "结束通话"}
        </button>
      </div>
    </div>
  );
}
