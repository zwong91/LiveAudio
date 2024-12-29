// src/app/api/rtc-connect/route.ts

import { NextRequest, NextResponse } from 'next/server';
import Cors from 'cors';

// 初始化 CORS 中间件
const cors = Cors({
  methods: ['POST', 'GET', 'HEAD'],
  origin: '*'  // 可以根据实际需求限制允许的源
});

function runMiddleware(req: NextRequest, res: NextResponse, fn: Function) {
  return new Promise((resolve, reject) => {
    fn(req, res, (result) => {
      if (result instanceof Error) {
        return reject(result);
      }
      return resolve(result);
    });
  });
}

const BASE_URL = "https://gtp.aleopool.cc/offer";

// 处理 OPTIONS 请求，用于 CORS 预检请求
export async function OPTIONS(req: NextRequest) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}

// 处理 POST 请求
export async function POST(req: NextRequest) {

  // 运行 CORS 中间件
  await runMiddleware(request, {}, cors);
  try {

    // 读取请求体，假设是 JSON 格式
    const body = await req.json();

    // 创建 URL 并设置查询参数
    //const url = new URL(BASE_URL);
    //url.searchParams.set('voice', 'ash');

    // 发送请求到外部 API
    const response = await fetch(BASE_URL, {
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

// 处理其他方法，返回 405 Method Not Allowed
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