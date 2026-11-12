/**
 * mat4.ts —— 最小 4x4 矩阵 / 四元数数学库（列主序，与 WebGL uniform 布局一致）
 * 数组下标 m[col * 4 + row]，即 m[12..14] 为平移 x/y/z。
 * 全部为纯函数，不引入任何第三方依赖。
 */

export type Mat4 = Float32Array; // 长度 16，列主序
export type Quat = Float32Array; // 长度 4：[x, y, z, w]
export type Vec3 = Float32Array; // 长度 3

export function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  return m;
}

export function mat4Copy(a: Mat4): Mat4 {
  return new Float32Array(a);
}

/** C = A * B（列主序标准矩阵乘法，变换顺序为先 B 后 A） */
export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + row] * b[col * 4 + k];
      }
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

/** 平移矩阵 */
export function mat4FromTranslation(x: number, y: number, z: number): Mat4 {
  const m = mat4Identity();
  m[12] = x;
  m[13] = y;
  m[14] = z;
  return m;
}

/** 由平移、四元数旋转、缩放组合为局部矩阵 T * R * S */
export function mat4FromTRS(t: Vec3 | number[], q: Quat, s: Vec3 | number[]): Mat4 {
  const [x, y, z, w] = q;
  const x2 = x + x;
  const y2 = y + y;
  const z2 = z + z;
  const xx = x * x2;
  const xy = x * y2;
  const xz = x * z2;
  const yy = y * y2;
  const yz = y * z2;
  const zz = z * z2;
  const wx = w * x2;
  const wy = w * y2;
  const wz = w * z2;

  const sx = s[0];
  const sy = s[1];
  const sz = s[2];

  const m = mat4Identity();
  m[0] = (1 - (yy + zz)) * sx;
  m[1] = (xy + wz) * sx;
  m[2] = (xz - wy) * sx;
  m[4] = (xy - wz) * sy;
  m[5] = (1 - (xx + zz)) * sy;
  m[6] = (yz + wx) * sy;
  m[8] = (xz + wy) * sz;
  m[9] = (yz - wx) * sz;
  m[10] = (1 - (xx + yy)) * sz;
  m[12] = t[0];
  m[13] = t[1];
  m[14] = t[2];
  return m;
}

export function quatIdentity(): Quat {
  return new Float32Array([0, 0, 0, 1]);
}

/** 绕任意单位轴旋转 angle（弧度） */
export function quatFromAxisAngle(axis: Vec3 | number[], angle: number): Quat {
  const half = angle * 0.5;
  const sin = Math.sin(half);
  return new Float32Array([axis[0] * sin, axis[1] * sin, axis[2] * sin, Math.cos(half)]);
}

/** 四元数球面线性插值；自动取最短弧 */
export function quatSlerp(a: Quat, b: Quat, t: number): Quat {
  let ax = a[0];
  let ay = a[1];
  let az = a[2];
  let aw = a[3];
  let bx = b[0];
  let by = b[1];
  let bz = b[2];
  let bw = b[3];

  let cosOmega = ax * bx + ay * by + az * bz + aw * bw;
  if (cosOmega < 0) {
    cosOmega = -cosOmega;
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
  }

  if (cosOmega > 1 - 1e-6) {
    // 夹角很小：退化为归一化线性插值，避免 sin 接近 0 时的数值问题
    const k0 = 1 - t;
    const k1 = t;
    const out = new Float32Array([
      k0 * ax + k1 * bx,
      k0 * ay + k1 * by,
      k0 * az + k1 * bz,
      k0 * aw + k1 * bw,
    ]);
    return quatNormalize(out);
  }

  const omega = Math.acos(Math.min(1, cosOmega));
  const sinOmega = Math.sin(omega);
  const k0 = Math.sin((1 - t) * omega) / sinOmega;
  const k1 = Math.sin(t * omega) / sinOmega;
  return new Float32Array([
    k0 * ax + k1 * bx,
    k0 * ay + k1 * by,
    k0 * az + k1 * bz,
    k0 * aw + k1 * bw,
  ]);
}

export function quatNormalize(q: Quat): Quat {
  const len = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return new Float32Array([q[0] / len, q[1] / len, q[2] / len, q[3] / len]);
}

export function vec3Lerp(a: Vec3 | number[], b: Vec3 | number[], t: number): Vec3 {
  return new Float32Array([
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ]);
}

/** 4x4 行列式（沿最后一行展开） */
export function mat4Determinant(m: Mat4): number {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  return (
    b00 * b11 - b01 * b10 + b02 * b09 +
    b03 * b08 - b04 * b07 + b05 * b06
  );
}

/**
 * 将仿射矩阵 M = T * R * S 分解为平移 / 旋转(四元数) / 缩放。
 * 输入为非奇异仿射矩阵（投影行必须为 [0,0,0,1]）。
 */
export function mat4Decompose(m: Mat4): { translation: Vec3; rotation: Quat; scale: Vec3 } {
  const translation: Vec3 = new Float32Array([m[12], m[13], m[14]]);

  // 三列去掉平移后即 R*S 的列向量，列长度即缩放
  const scaleX = Math.hypot(m[0], m[1], m[2]);
  const scaleY = Math.hypot(m[4], m[5], m[6]);
  const scaleZ = Math.hypot(m[8], m[9], m[10]);
  if (scaleX < 1e-8 || scaleY < 1e-8 || scaleZ < 1e-8) {
    throw new Error('mat4Decompose 失败：矩阵为奇异矩阵，存在零长度缩放轴，无法分解');
  }

  // 归一化列得到纯旋转矩阵
  const r = new Float32Array(16);
  r[0] = m[0] / scaleX;
  r[1] = m[1] / scaleX;
  r[2] = m[2] / scaleX;
  r[4] = m[4] / scaleY;
  r[5] = m[5] / scaleY;
  r[6] = m[6] / scaleY;
  r[8] = m[8] / scaleZ;
  r[9] = m[9] / scaleZ;
  r[10] = m[10] / scaleZ;
  r[15] = 1;

  // 处理含负缩放导致的反射：若行列式为负，翻转一个轴
  let det =
    r[0] * (r[5] * r[10] - r[6] * r[9]) -
    r[4] * (r[1] * r[10] - r[2] * r[9]) +
    r[8] * (r[1] * r[6] - r[2] * r[5]);
  let sx = scaleX;
  let sy = scaleY;
  let sz = scaleZ;
  if (det < 0) {
    // 翻转最小缩放轴，保证旋转矩阵为正常旋转
    if (sx <= sy && sx <= sz) {
      sx = -sx;
      r[0] = -r[0]; r[1] = -r[1]; r[2] = -r[2];
    } else if (sy <= sz) {
      sy = -sy;
      r[4] = -r[4]; r[5] = -r[5]; r[6] = -r[6];
    } else {
      sz = -sz;
      r[8] = -r[8]; r[9] = -r[9]; r[10] = -r[10];
    }
  }

  // 旋转矩阵 -> 四元数（Shepperd 方法，数值稳定）
  const trace = r[0] + r[5] + r[10];
  let q: Quat;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    q = new Float32Array([
      (r[6] - r[9]) / s,
      (r[8] - r[2]) / s,
      (r[1] - r[4]) / s,
      0.25 * s,
    ]);
  } else if (r[0] > r[5] && r[0] > r[10]) {
    const s = Math.sqrt(1 + r[0] - r[5] - r[10]) * 2;
    q = new Float32Array([
      0.25 * s,
      (r[4] + r[1]) / s,
      (r[8] + r[2]) / s,
      (r[6] - r[9]) / s,
    ]);
  } else if (r[5] > r[10]) {
    const s = Math.sqrt(1 + r[5] - r[0] - r[10]) * 2;
    q = new Float32Array([
      (r[4] + r[1]) / s,
      0.25 * s,
      (r[9] + r[6]) / s,
      (r[8] - r[2]) / s,
    ]);
  } else {
    const s = Math.sqrt(1 + r[10] - r[0] - r[5]) * 2;
    q = new Float32Array([
      (r[8] + r[2]) / s,
      (r[9] + r[6]) / s,
      0.25 * s,
      (r[1] - r[4]) / s,
    ]);
  }

  return {
    translation,
    rotation: quatNormalize(q),
    scale: new Float32Array([sx, sy, sz]),
  };
}

/** 求逆（通用 4x4，伴随矩阵法） */
export function mat4Invert(a: Mat4): Mat4 {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det =
    b00 * b11 - b01 * b10 + b02 * b09 +
    b03 * b08 - b04 * b07 + b05 * b06;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-10) {
    throw new Error('mat4Invert 失败：矩阵奇异或接近奇异，无法求逆（det=' + det + '）');
  }
  det = 1.0 / det;

  const out = new Float32Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

/** 4x4 矩阵 × 点 w=1，返回 vec3 平移后的坐标 */
export function mat4TransformPoint(m: Mat4, p: Vec3 | number[]): Vec3 {
  const x = p[0];
  const y = p[1];
  const z = p[2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  const invW = w !== 0 ? 1 / w : 1;
  return new Float32Array([
    (m[0] * x + m[4] * y + m[8] * z + m[12]) * invW,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) * invW,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) * invW,
  ]);
}

/** 通用线性插值（用于缩放/权重等标量向量） */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

/** 格式化矩阵用于控制台输出 */
export function mat4ToString(m: Mat4, precision = 4): string {
  const f = (v: number) => {
    const s = (Math.abs(v) < 1e-10 ? 0 : v).toFixed(precision);
    return s.padStart(8);
  };
  const rows: string[] = [];
  for (let row = 0; row < 4; row++) {
    rows.push(`  [${f(m[row])} ${f(m[4 + row])} ${f(m[8 + row])} ${f(m[12 + row])}]`);
  }
  return rows.join('\n');
}
