// src/app/api/rtc-connect/route.ts
export const runtime = 'edge';

import { NextRequest, NextResponse } from 'next/server';

//const BASE_URL = "https://audio.enty.services/offer";
const BASE_URL = "https://gtp.aleopool.cc/offer";

// 自定义 CORS 中间件
function handleCors(req: NextRequest, res: NextResponse) {
  // 允许来自所有源的请求
  res.headers.set('Access-Control-Allow-Origin', '*');
  res.headers.set('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.headers.set('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    // 如果是 OPTIONS 请求，直接返回 200
    return new NextResponse(null, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
      },
    });
  }
  return res;
}

export async function OPTIONS(req: NextRequest) {
  const res = NextResponse.next(); // 创建默认响应对象
  return handleCors(req, res); // 处理 CORS 逻辑
}

export async function POST(req: NextRequest) {
  const res = NextResponse.next(); // 创建默认响应对象
  handleCors(req, res); // 处理 CORS 逻辑

  try {
    console.log('Received POST request', req);
    const body = await req.json();
    console.log('Request Body:', body);
    // 发送请求到外部 API
    const response = await fetch(BASE_URL, {
      method: 'POST',
      body: JSON.stringify(body),
      headers: {
        'Content-Type': 'application/json',
      },
    });
    console.log('External API Response:', response);
    // 检查响应是否正常
    if (!response.ok) {
      return new NextResponse('WebRTC API error', { status: response.status });
    }

    const jsonResponse = await response.json();

    return new NextResponse(JSON.stringify(jsonResponse), {
      headers: {
        'Content-Type': 'application/json',
      },
    });
  } catch (error) {
    return new NextResponse(`Error: ${(error as Error).message}`, { status: 500 });
  }
}

export async function GET(req: NextRequest) {
  return new NextResponse('Method Not Allowed', { status: 405 });
}

export async function PUT(req: NextRequest) {
  return new NextResponse('Method Not Allowed', { status: 405 });
}

export async function DELETE(req: NextRequest) {
  return new NextResponse('Method Not Allowed', { status: 405 });
}

export async function PATCH(req: NextRequest) {
  return new NextResponse('Method Not Allowed', { status: 405 });
}

export async function HEAD(req: NextRequest) {
  return new NextResponse('Method Not Allowed', { status: 405 });
}
