import { ContactShadows, Html, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { useThemeVersion } from "../hooks";
import modelUrl from "../assets/human.glb?inline";
import organsUrl from "../assets/organs.glb?inline";
import organsMeta from "../assets/organs.json";
import type { Level } from "./ui";

/**
 * The virtual patient in 3D: the MakeHuman base mesh (CC0), shaped by the patient's sex and BMI
 * through morph targets, shown as a translucent clinical body with the anatomical organs the twin
 * models inside it (BodyParts3D, CC BY-SA 2.1 JP), each coloured by the twin's estimate for it.
 * Built by scripts/build_body_model.py and scripts/build_organ_models.py.
 */

export interface Organ3D {
  key: string;
  name: string;
  value: string;
  level: Level;
}
type V3 = [number, number, number];

// The model and its landmarks ship inside this lazily loaded chunk and are decoded in memory:
// no extra request, and a load that never depends on the network.
function parseGlb(dataUrl: string) {
  const bin = atob(dataUrl.slice(dataUrl.indexOf(",") + 1));
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new GLTFLoader().parseAsync(buf.buffer, "");
}

let bodyPromise: Promise<THREE.Mesh> | null = null;
function loadBody(): Promise<THREE.Mesh> {
  if (!bodyPromise) {
    bodyPromise = parseGlb(modelUrl).then((gltf) => {
      let mesh: THREE.Mesh | null = null;
      gltf.scene.traverse((o) => {
        if ((o as THREE.Mesh).isMesh && !mesh) mesh = o as THREE.Mesh;
      });
      if (!mesh) throw new Error("no mesh in body model");
      return mesh;
    });
  }
  return bodyPromise;
}

function useBodyMesh(): THREE.Mesh | null {
  const [m, setM] = useState<THREE.Mesh | null>(null);
  useEffect(() => {
    let live = true;
    loadBody().then((src) => {
      if (!live) return;
      const c = src.clone();
      c.geometry = src.geometry.clone();
      setM(c);
    }).catch((e) => console.error("3D body failed to load", e));
    return () => { live = false; };
  }, []);
  return m;
}
const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const LEVEL_VAR: Record<Level, string> = { good: "--good", warning: "--warning", serious: "--serious", critical: "--critical" };

function Body({ sex, bmi, xray }: { sex: string | null | undefined; bmi: number | null | undefined; xray: boolean }) {
  const mesh = useBodyMesh();
  const theme = useThemeVersion();

  const material = useMemo(() => {
    const dark = document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const m = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(dark ? "#7f97b8" : "#c9d6e8"), roughness: 0.35, metalness: 0, clearcoat: 0.5,
      transparent: true, opacity: xray ? 0.16 : 0.92, depthWrite: !xray, side: THREE.FrontSide,
    });
    const rim = new THREE.Color(css("--series-1") || "#2a78d6");
    m.onBeforeCompile = (shader) => {
      shader.uniforms.uRim = { value: rim };
      shader.uniforms.uRimAlpha = { value: xray ? 0.55 : 0.0 };
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", "#include <common>\nuniform vec3 uRim;\nuniform float uRimAlpha;")
        .replace("#include <emissivemap_fragment>", `#include <emissivemap_fragment>
          float fres = pow(1.0 - abs(dot(normal, normalize(vViewPosition))), 2.4);
          totalEmissiveRadiance += uRim * fres * 0.9;
          diffuseColor.a = clamp(diffuseColor.a + fres * uRimAlpha, 0.0, 1.0);`);
    };
    m.customProgramCacheKey = () => `body-${xray}`;
    return m;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [xray, theme]);

  // shape the twin like the patient: sex, and weight from BMI. The MakeHuman morph targets are
  // blended once on the CPU and the normals recomputed, so lighting is right for every body shape.
  useEffect(() => {
    if (!mesh) return;
    const g = mesh.geometry;
    const dict = mesh.morphTargetDictionary ?? {};
    const morphs = g.morphAttributes.position;
    if (!morphs) return;
    if (!g.userData.base) g.userData.base = (g.attributes.position.array as Float32Array).slice();
    const base = g.userData.base as Float32Array;
    const sexes: ["male" | "female", number][] = sex === "F" ? [["female", 1]] : sex === "M" ? [["male", 1]] : [["male", 0.5], ["female", 0.5]];
    const b = bmi ?? 23;
    const w = b >= 23 ? Math.min((b - 23) / 12, 1) : -Math.min((23 - b) / 5, 1);
    const weights: [string, number][] = [];
    for (const [sx, k] of sexes) {
      weights.push([sx, k]);
      if (w > 0) weights.push([`${sx}_heavy`, k * w]);
      if (w < 0) weights.push([`${sx}_light`, -k * w]);
    }
    const out = base.slice();
    for (const [name, k] of weights) {
      const d = morphs[dict[name]]?.array as Float32Array | undefined;
      if (d) for (let i = 0; i < out.length; i++) out[i] += k * d[i];
    }
    g.setAttribute("position", new THREE.BufferAttribute(out, 3));
    g.computeVertexNormals();
    g.computeBoundingSphere();
    mesh.morphTargetInfluences?.fill(0);
  }, [mesh, sex, bmi]);

  return mesh ? <primitive object={mesh} material={material} renderOrder={2} /> : null;
}

interface OrganFit { scale: number; offset: V3 }
const ORGAN_FIT = (organsMeta as unknown as { fit: Record<"male" | "female", OrganFit> }).fit;
// natural tissue tints for organs the twin has no reading for (status colours replace them otherwise)
const TINT: Record<string, string> = { heart: "#b5443f", liver: "#8e3b2e", pancreas: "#dcae79", gut: "#d98f8a", kidney: "#9a3a3a", muscle: "#bf4f4a" };

let organsPromise: Promise<Record<string, THREE.Mesh[]>> | null = null;
/** BodyParts3D organs (CC BY-SA 2.1 JP), grouped by the twin's organ key; "ctx" = lungs and aorta for context. */
function loadOrgans(): Promise<Record<string, THREE.Mesh[]>> {
  if (!organsPromise) {
    organsPromise = parseGlb(organsUrl).then((gltf) => {
      const out: Record<string, THREE.Mesh[]> = {};
      gltf.scene.updateMatrixWorld(true);
      gltf.scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (!m.isMesh) return;
        const name = (m.name || m.parent?.name || "").toLowerCase();
        const key = name.split("_")[0];
        m.geometry.applyMatrix4(m.matrixWorld); // bake node transforms: the organs share one frame
        m.geometry.computeBoundingBox();
        (out[key] ??= []).push(m);
      });
      return out;
    });
  }
  return organsPromise;
}

function useOrgans() {
  const [o, setO] = useState<Record<string, THREE.Mesh[]> | null>(null);
  useEffect(() => {
    let live = true;
    loadOrgans().then((x) => live && setO(x)).catch((e) => console.error("3D organs failed to load", e));
    return () => { live = false; };
  }, []);
  return o;
}

function boxOf(meshes: THREE.Mesh[]): THREE.Box3 {
  const b = new THREE.Box3();
  meshes.forEach((m) => m.geometry.boundingBox && b.union(m.geometry.boundingBox));
  return b;
}

/** One anatomical organ (possibly several meshes), coloured by the twin's status; the heart beats. */
function Organ({ k, meshes, o, active, onHover, bpm }: {
  k: string; meshes: THREE.Mesh[]; o: Organ3D | undefined; active: boolean; onHover: (k: string | null) => void; bpm: number | null;
}) {
  const theme = useThemeVersion();
  const box = useMemo(() => boxOf(meshes), [meshes]);
  const center = useMemo(() => box.getCenter(new THREE.Vector3()), [box]);
  const material = useMemo(() => {
    const c = new THREE.Color(o ? css(LEVEL_VAR[o.level]) || TINT[k] : TINT[k] ?? "#cc8888");
    return new THREE.MeshPhysicalMaterial({ color: c, emissive: c, emissiveIntensity: active ? 0.55 : o ? 0.18 : 0.04,
      roughness: 0.42, metalness: 0, clearcoat: 0.7, clearcoatRoughness: 0.35, sheen: 0.4, sheenColor: new THREE.Color("#ffffff") });
  }, [o, k, active, theme]); // eslint-disable-line react-hooks/exhaustive-deps
  const pulse = useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    if (k !== "heart" || !pulse.current) return;
    const ph = (clock.elapsedTime * ((bpm ?? 64) / 60)) % 1; // the patient's own night heart rate
    const beat = Math.exp(-((ph - 0.08) ** 2) / 0.002) + 0.5 * Math.exp(-((ph - 0.28) ** 2) / 0.003);
    pulse.current.scale.setScalar(1 + 0.05 * beat);
  });
  const size = box.getSize(new THREE.Vector3());
  const side = center.x >= 0 ? 1 : -1;
  return (
    <group onPointerOver={(e) => { e.stopPropagation(); onHover(k); }} onPointerOut={() => onHover(null)}>
      <group ref={pulse} position={center}>
        {meshes.map((m) => (
          <mesh key={m.uuid} geometry={m.geometry} material={material} position={[-center.x, -center.y, -center.z]} renderOrder={1} />
        ))}
      </group>
      {active && o && (
        <Html position={[center.x + side * (size.x / 2 + 0.06), center.y + size.y * 0.25, center.z]} center zIndexRange={[20, 0]} style={{ pointerEvents: "none" }}>
          <div className="whitespace-nowrap rounded-md border px-2 py-1 text-[12px] shadow-sm" style={{ background: "var(--surface)", borderColor: "var(--border)", color: "var(--ink)" }}>
            <div className="font-medium">{o.name.split(" · ")[0]}</div>
            <div style={{ color: "var(--ink-2)" }}>{o.value}</div>
          </div>
        </Html>
      )}
    </group>
  );
}

/** The anatomical organ set, scaled and placed into this patient's body shape. */
function Organs({ organs, sex, active, onHover, bpm }: {
  organs: Organ3D[]; sex: string | null | undefined; active: string | null; onHover: (k: string | null) => void; bpm: number | null;
}) {
  const parts = useOrgans();
  const w = sex === "F" ? 1 : sex === "M" ? 0 : 0.5;
  const f = ORGAN_FIT.male, g = ORGAN_FIT.female;
  const scale = f.scale * (1 - w) + g.scale * w;
  const offset = f.offset.map((x, i) => x * (1 - w) + g.offset[i] * w) as V3;
  const ctxMaterial = useMemo(() => new THREE.MeshPhysicalMaterial({ color: new THREE.Color("#d9a7a7"), transparent: true, opacity: 0.16,
    roughness: 0.6, depthWrite: false }), []);
  if (!parts) return null;
  const byKey = Object.fromEntries(organs.map((o) => [o.key, o]));
  return (
    <group position={offset} scale={scale}>
      {Object.entries(parts).filter(([key]) => key !== "ctx").map(([key, meshes]) => (
        <Organ key={key} k={key} meshes={meshes} o={byKey[key]} active={active === key} onHover={onHover} bpm={bpm} />
      ))}
      {(parts.ctx ?? []).map((m) => <mesh key={m.uuid} geometry={m.geometry} material={ctxMaterial} renderOrder={0} />)}
    </group>
  );
}

const VIEWS = {
  body: { target: new THREE.Vector3(0, 0.84, 0), dist: 3.9 },
  organs: { target: new THREE.Vector3(0, 1.08, 0), dist: 1.35 },
};

/** Eases the camera between the full-body and organ views without fighting the user's own rotation. */
function CameraRig({ view, controls }: { view: keyof typeof VIEWS; controls: React.RefObject<{ target: THREE.Vector3; update: () => void } | null> }) {
  const { camera } = useThree();
  const goal = useRef<{ target: THREE.Vector3; dist: number } | null>(null);
  useEffect(() => {
    goal.current = VIEWS[view];
  }, [view]);
  useFrame(() => {
    const c = controls.current;
    if (!goal.current || !c) return;
    c.target.lerp(goal.current.target, 0.12);
    const dir = camera.position.clone().sub(c.target).normalize();
    const want = c.target.clone().add(dir.multiplyScalar(goal.current.dist));
    camera.position.lerp(want, 0.12);
    c.update();
    if (camera.position.distanceTo(want) < 0.005 && c.target.distanceTo(goal.current.target) < 0.002) goal.current = null;
  });
  return null;
}

/** Exposes render progress on the container (data-state), for tests and headless screenshots. */
function Progress({ host }: { host: React.RefObject<HTMLDivElement> }) {
  const n = useRef(0);
  useFrame(() => {
    n.current += 1;
    if (host.current && (n.current <= 3 || n.current % 30 === 0)) host.current.dataset.frames = String(n.current);
  });
  return null;
}

export default function Body3D({ organs, sex, bmi, bpm, active, onHover }: {
  organs: Organ3D[]; sex: string | null | undefined; bmi: number | null | undefined; bpm: number | null;
  active: string | null; onHover: (k: string | null) => void;
}) {
  const [view, setView] = useState<keyof typeof VIEWS>("body");
  const [xray, setXray] = useState(true);
  const [interacting, setInteracting] = useState(false);
  const controls = useRef<{ target: THREE.Vector3; update: () => void } | null>(null);
  const reduced = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const host = useRef<HTMLDivElement>(null);
  // ?debug3d in the URL shows the render status on screen (WebGL context, model load, frames)
  const debug = typeof window !== "undefined" && window.location.href.includes("debug3d");
  const [dbg, setDbg] = useState("");
  useEffect(() => {
    if (!debug) return;
    const id = window.setInterval(() => {
      const d = host.current?.dataset ?? {};
      setDbg(`state=${d.state ?? "-"} gl=${d.gl ?? "-"} frames=${d.frames ?? "0"} t=${(performance.now() / 1000).toFixed(1)}s`);
    }, 250);
    return () => window.clearInterval(id);
  }, [debug]);
  // react-three-fiber creates the renderer only after measuring its container; some environments
  // (background tabs, headless browsers) never deliver that first measurement, so nudge it
  useEffect(() => {
    const ids = [150, 500, 1200, 2500, 5000].map((ms) => window.setTimeout(() => {
      if (!host.current?.dataset.gl) window.dispatchEvent(new Event("resize"));
    }, ms));
    return () => ids.forEach((id) => window.clearTimeout(id));
  }, []);
  useEffect(() => {
    if (host.current) host.current.dataset.state = "mounted";
    loadBody().then(() => host.current && (host.current.dataset.state = "model-ready")).catch((e) => host.current && (host.current.dataset.state = `error: ${e}`));
  }, []);

  return (
    <div className="relative h-full w-full" ref={host} data-testid="body3d">
      {debug && <div className="absolute bottom-0 left-0 z-10 bg-black/70 px-1 font-mono text-[10px] text-white">{dbg}</div>}
      <Canvas onCreated={() => host.current && (host.current.dataset.gl = "ok")} camera={{ position: [0.7, 0.95, 3.85], fov: 30, near: 0.05, far: 30 }} dpr={[1, 1.5]} gl={{ antialias: true, alpha: true, preserveDrawingBuffer: true }}
        onPointerDown={() => setInteracting(true)} onPointerUp={() => setInteracting(false)}
        aria-label="3D virtual patient: drag to rotate, scroll to zoom">
        <ambientLight intensity={0.7} />
        <directionalLight position={[2, 3, 3]} intensity={1.4} />
        <directionalLight position={[-3, 2, -2]} intensity={0.8} />
        <Suspense fallback={null}>
          <Body sex={sex} bmi={bmi} xray={xray} />
          {xray && <Organs organs={organs} sex={sex} active={active} onHover={onHover} bpm={bpm} />}
          <ContactShadows position={[0, 0.001, 0]} opacity={0.3} scale={1.6} blur={2.4} far={1.2} frames={30} resolution={256} />
        </Suspense>
        <OrbitControls ref={controls as never} target={[0, 0.84, 0]} enablePan={false} minDistance={0.9} maxDistance={4.5}
          minPolarAngle={0.35} maxPolarAngle={1.75} autoRotate={!reduced && !interacting && active == null} autoRotateSpeed={0.7} />
        <CameraRig view={view} controls={controls} />
        <Progress host={host} />
      </Canvas>
      <div className="absolute left-1 top-1 flex gap-1 text-[11px]" role="group" aria-label="3D view">
        {(["body", "organs"] as const).map((v) => (
          <button key={v} className="btn px-2 py-0.5" aria-pressed={view === v} onClick={() => setView(v)}
            style={view === v ? { borderColor: "var(--series-1)", color: "var(--series-1)" } : undefined}>{v === "body" ? "Full body" : "Organs"}</button>
        ))}
        <button className="btn px-2 py-0.5" aria-pressed={!xray} onClick={() => setXray((x) => !x)}>{xray ? "Skin" : "X-ray"}</button>
      </div>
    </div>
  );
}

loadBody();
loadOrgans();
