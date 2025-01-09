"use client";

import { useEffect, useState, useRef } from "react";
import styles from "./page.module.css";
import { useMicVAD, utils } from "@ricky0123/vad-react"

import LanguageSelection from './languageselect';

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
  checkAndBufferAudio: Function,
  isSimultaneous: boolean,
  targetLang: string
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

      // 从环境变量中获取值
      const username = process.env.NEXT_PUBLIC_USERNAME;
      const credential = process.env.NEXT_PUBLIC_CREDENTIAL;
      // Check if environment variables are set
      if (!username || !credential) {
        console.error("TURN server credentials are missing!");
        return;
      }
      // ICE 服务器配置
      const iceServers = [
        // {
        //   urls: [
        //     "stun:stun.l.google.com:19302",    // Google STUN 服务器
        //     "stun:stun1.l.google.com:19302",   // 备用 Google STUN 服务器
        //     "stun:stun2.l.google.com:19302",   // 备用 Google STUN 服务器
        //     "stun:stun3.l.google.com:19302",   // 备用 Google STUN 服务器
        //     "stun:stun4.l.google.com:19302"    // 备用 Google STUN 服务器
        //     "stun:stun.cloudflare.com:3478",   // cf turn service
        //   ]
        // },

        // // cf turn service
        // { urls: 'stun:stun.cloudflare.com:3478' },
        // {
        //   "urls":[
        //     "turn:turn.cloudflare.com:3478?transport=udp",
        //     "turn:turn.cloudflare.com:3478?transport=tcp",
        //     "turns:turn.cloudflare.com:5349?transport=tcp"
        //   ],
        //   "username":"g024564e46fd561d7728d9b2170add06f0907dba749880ec5a47eee1422629b1",
        //   "credential":"b02671af21caf757a82c739db804f4309a4551ceb8962bab57760ff8c5536d9c"
        // },

        { urls: 'stun:gtp.aleopool.cc:3478' },
        //如果需要，可以添加 TURN 服务器
        {
          urls: "turn:gtp.aleopool.cc:3478",   // TURN 服务器
          username: username,            // TURN 服务器用户名
          credential: credential, // TURN 服务器密码
          credentialType: 'password',
          realm: 'gtp.aleopool.cc',
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
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
          // 添加音频轨道到 PeerConnection
          stream.getTracks().forEach((track) => {
            console.log("Adding track to connection:", track);
            //pc.addTransceiver(track, { direction: "sendrecv" });
            pc.addTrack(track)
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
      const dc = pc.createDataChannel('c');
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
      peerConnection.onicecandidate = (event: RTCPeerConnectionIceEvent) => {
        if (event.candidate) {
          console.log('获取到ICE候选:', event.candidate.type, event.candidate.address);
          if (event.candidate.type === 'srflx') {
            console.log(`STUN成功！ 公网IP: ${event.candidate.address}, 公网端口: ${event.candidate.port}`);
          }
          if (event.candidate.type === 'relay') {
            console.log(`TURN成功！ 中继IP: ${event.candidate.address}, 中继端口: ${event.candidate.port}`);
          }
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
          console.log("DataChannel opened and ready to use:", dataChannel.label);
          const audioConfig = {
            type: 'config',
            data: {
                is_simultaneous: isSimultaneous,
                target_lang: targetLang,
            }
        };
        // 发送配置数据
        if (dataChannel.readyState === 'open') {
          dataChannel.send(JSON.stringify(audioConfig));
        } else {
          console.error("DataChannel is not open, unable to send data.");
        }
        
        const pingInterval = setInterval(() => {
            if (dataChannel.readyState === 'open') {
              //dataChannel.send(JSON.stringify({ type: "ping" }));
              dataChannel.send(JSON.stringify(audioConfig));
            } else {
              console.error("DataChannel is not open, unable to send data.");
            }
          }, 5000);
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
            console.error("Error processing WebRTC message:", error);
          }
        };

        dataChannel.onclose = () => {
          console.log("DataChannel closed:", dataChannel.label);
        };
      };
  
    }
  }, [peerConnection, dataChannel, isSimultaneous, targetLang, checkAndBufferAudio]);  

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
    peerConnection,
    dataChannel,
  };
};


// 主组件
export default function Home() {
  const [audioQueue, setAudioQueue] = useState<Blob[]>([]);
  const [isRecording, setIsRecording] = useState(true);
  const [audioList, setAudioList] = useState<string[]>([]);

  const [isSimultaneous, setIsSimultaneous] = useState(false);
  const [targetLang, setTargetLang] = useState('英语');
  const handleLanguageChange = (newIsSimultaneous: boolean, newTargetLang: string) => {
    setIsSimultaneous(newIsSimultaneous);
    setTargetLang(newTargetLang);
    console.log('Updated Language Config:', newIsSimultaneous, newTargetLang);

  // 在语言变化后触发发送配置数据
  if (dataChannel && dataChannel.readyState === 'open') {
    const audioConfig = {
      type: 'config',
      data: {
        is_simultaneous: newIsSimultaneous,
        target_lang: newTargetLang,
      },
    };
    dataChannel.send(JSON.stringify(audioConfig));
    console.log('Language config sent:', audioConfig);
  } else {
    console.error("DataChannel is not open, unable to send data.");
  }

  };

  // const audioItemKey = (audioURL: string) => audioURL.substring(-10)
  // const vad = useMicVAD({
  //   model: "v5",
  //   baseAssetPath: "/",
  //   onnxWASMBasePath: "/",
  //   onSpeechEnd: (audio: Float32Array) => {
  //     const wavBuffer = utils.encodeWAV(audio);
  //     const base64 = utils.arrayBufferToBase64(wavBuffer);
  //     const url = `data:audio/wav;base64,${base64}`;
  //     setAudioList((old) => [url, ...old]);
  //   },
  // });

  const { isPlayingAudio, playAudio, checkAndBufferAudio, stopCurrentAudio } = useAudioManager(
    audioQueue,
    setAudioQueue,
    setIsRecording
  );
  const { connectionStatus, isCallEnded, endCall, peerConnection, dataChannel } = useWebRTC(
    audioQueue,
    setAudioQueue,
    setIsRecording,
    checkAndBufferAudio,
    isSimultaneous,
    targetLang
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

  // Add wake lock logic
  useEffect(() => {
    let wakeLock: WakeLockSentinel | null = null;

    // Request screen wake lock to prevent the screen from going to sleep
    const requestWakeLock = async () => {
      try {
        wakeLock = await navigator.wakeLock.request("screen");
        console.log("Screen wake lock acquired");
      } catch (error) {
        console.error("Error with requestWakeLock", error);
      }
    };

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
              ? "AI Offline"
              : isPlayingAudio
              ? "AI is Speaking"
              : "AI is Listening"}
            </span>
          </div>
        </div>
      </div>

      <div className={styles.langContent}>
        <LanguageSelection
          onLanguageChange={handleLanguageChange} // Pass the language change handler
        />
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
          {isCallEnded ? "Call Again" : "End Call"}
        </button>
      </div>
    </div>
  );
}
