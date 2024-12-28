// src/app/api/rtc-connect/route.ts

import { NextResponse } from 'next/server';

const DEFAULT_INSTRUCTIONS = `You are helpful and have some tools installed.

In the tools you have the ability to control a robot hand.
`;

const BASE_URL = "https://gtp.aleopool.cc/offer";

export async function POST(req: Request) {
  try {
    // 读取请求体，假设是 JSON 格式
    const body = await req.json();

    // 创建 URL 并设置查询参数
    const url = new URL(BASE_URL);
    url.searchParams.set('voice', 'ash');

    // 发送请求到外部 API
    const response = await fetch(url.toString(), {
      method: 'POST',
      body: JSON.stringify(body), // 确保请求体是 JSON 格式
      headers: {
        'Content-Type': 'application/json', // 请求头设置为 JSON 格式
      },
    });

    // 检查响应是否正常
    if (!response.ok) {
      return new NextResponse('WebRTC API error', { status: response.status });
    }

    // 获取外部 API 的 JSON 响应
    const jsonResponse = await response.json();

    // 返回外部 API 响应的 JSON 数据
    return new NextResponse(JSON.stringify(jsonResponse), {
      headers: {
        'Content-Type': 'application/json', // 响应内容是 JSON 格式
      },
    });
  } catch (error) {
    return new NextResponse(`Error: ${(error as Error).message}`, { status: 500 });
  }
}
