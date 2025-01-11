"use client";

import { useEffect, useState, useRef } from "react";
import styles from "./page.module.css";
import { useMicVAD, utils } from "@ricky0123/vad-react"

import LanguageSelection from './languageselect';
import { WavRecorder, WavStreamPlayer } from 'wavtools-patch';
import { json } from "stream/consumers";

const wavStreamPlayer = new WavStreamPlayer({ sampleRate: 16000 });

// 请求麦克风权限
const requestMicrophonePermission = async () => {
  try {
    //ios 确保在用户交互下请求权限
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    stream.getTracks().forEach(track => track.stop()); // 停止音频流，释放资源
  } catch (error) {
    alert("麦克风权限被拒绝，无法进行 WebRTC 通信");
    return;
  }
};

// 音频管理器
const useAudioManager = (setIsPlayingAudio: Function, setIsRecording: Function) => {
  const checkAndBufferAudio = (audioData: ArrayBuffer) => {
    //const audio = new Int16Array(audioData);

    // Queue 3s of audio, will start playing immediately
    //wavStreamPlayer.add16BitPCM(chunk.data, chunk.track);
    // json: {
      // track: str "my-track"
      // mimeType: str "pcm16"
      // data: bytes Int16Array
    //}
    wavStreamPlayer.add16BitPCM(audioData, 'my-track');
    setIsPlayingAudio(true)
    // get data for visualization
    const frequencyData = wavStreamPlayer.getFrequencies();

    // const text = new TextDecoder("utf-8").decode(audioData);

    // if (text.includes("END_OF_AUDIO")) {
    //   console.log("Detected END_OF_AUDIO signal in audioData");
    //   stopCurrentAudio(); // 停止当前音频播放
    //   setIsRecording(true);
    //   setIsPlayingAudio(false);
    //   return;
    // }

    // // 如果没有检测到 "END_OF_AUDIO" 信号，继续缓存音频并立即播放
  };

  return {
    checkAndBufferAudio,
  };
};

// WebRTC 管理器
const useWebRTC = (
  audioQueue: Blob[],
  setAudioQueue: Function,
  setIsRecording: Function,
  checkAndBufferAudio: Function,
  setIsPlayingAudio: Function,
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
      const dc = pc.createDataChannel('c-events');
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

    // Attach event listeners to the data channel when a new one is created
  useEffect(() => {
    if (dataChannel) {
      // Append new server events to the list
      dataChannel.addEventListener("message", async (e) => {
        console.log("c-events channel received message:", e.data);
        try {
          // 解析 JSON 数据
          const json = JSON.parse(e.data);
          //console.log("Parsed JSON:", json);
        } catch (error) {
          console.error("Error processing WebRTC message:", error);
        }
      });

      // Set session active when the data channel is opened
      dataChannel.addEventListener("open", () => {
        console.log("DataChannel opened and ready to use:", dataChannel.label);
        const pingInterval = setInterval(() => {
          if (dataChannel.readyState === 'open') {
            dataChannel.send(JSON.stringify({ type: "ping" }));
          } else {
            console.error("DataChannel is not open, unable to send data.");
          }
        }, 5000);

      });

      // Handle the close event
      dataChannel.addEventListener("close", () => {
        console.log("DataChannel has been closed:", dataChannel.label);
        // Perform cleanup or additional logic here
      });

    }
  }, [dataChannel, isSimultaneous, targetLang, checkAndBufferAudio]);
  
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
        dataChannel.onopen = async () => {
          console.log("DataChannel opened and ready to use:", dataChannel.label);
          // Connect to microphone
          await wavStreamPlayer.connect();
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

        wavStreamPlayer.onStop = () => setIsPlayingAudio(false);
        
        const pingInterval = setInterval(() => {
            if (dataChannel.readyState === 'open') {
              dataChannel.send(JSON.stringify({ type: "ping" }));
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

        dataChannel.onclose = async () => {
          console.log("DataChannel closed:", dataChannel.label);
          // Interrupt the audio (halt playback) at any time
          // To restart, need to call .add16BitPCM() again
          const trackOffset = wavStreamPlayer.interrupt();
          console.log(`Track ID: ${trackOffset.trackId}, sample number: ${trackOffset.offset}, time in track: ${trackOffset.currentTime}`);
        };
      };
  
    }
  }, [peerConnection, isSimultaneous, targetLang, checkAndBufferAudio]);  

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
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioList, setAudioList] = useState<string[]>([]);
  const [hasPermission, setHasPermission] = useState(false);
  const [isSimultaneous, setIsSimultaneous] = useState(false);
  const [targetLang, setTargetLang] = useState('中文');
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

  const {checkAndBufferAudio } = useAudioManager(
    setIsPlayingAudio,
    setIsRecording
  );
  const { connectionStatus, isCallEnded, endCall, peerConnection, dataChannel } = useWebRTC(
    audioQueue,
    setAudioQueue,
    setIsRecording,
    checkAndBufferAudio,
    setIsPlayingAudio,
    isSimultaneous,
    targetLang
  );

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

// 处理开始通话的逻辑，确保在用户交互下请求权限
const startCall = () => {
  requestMicrophonePermission()
    .then(() => {
      // 在此处继续执行 WebRTC 相关的代码
      setHasPermission(true);
    })
    .catch(() => {
      console.error("Failed to get microphone permission");
    });
};

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

// <div>
// {/* Add the VAD status */}
// <div>
//   <h6>Listening</h6>
//   {!vad.listening && "Not"} listening
//   <h6>Loading</h6>
//   {!vad.loading && "Not"} loading
//   <h6>Errored</h6>
//   {!vad.errored && "Not"} errored
//   <h6>User Speaking</h6>
//   {!vad.userSpeaking && "Not"} speaking
//   <h6>Audio count</h6>
//   {audioList.length}
//   <h6>Start/Pause</h6>
//   <button onClick={vad.pause}>Pause</button>
//   <button onClick={vad.start}>Start</button>
//   <button onClick={vad.toggle}>Toggle</button>
// </div>

// {/* Add the audio playlist */}
// <div>
//   <ol
//     id="playlist"
//     className="self-center pl-0 max-h-[400px] overflow-y-auto no-scrollbar list-none"
//   >
//     {audioList.map((audioURL) => {
//       return (
//         <li className="pl-0" key={audioItemKey(audioURL)}>
//           <audio src={audioURL} controls />
//         </li>
//       );
//     })}
//   </ol>
// </div>
// </div>
