import { Canvas } from '@react-three/fiber'

export default function ChartContext({ width = 400, height = 200, children }) {
  return (
    <Canvas
      style={{ width, height }}
      onCreated={({ camera, gl }) => {
        gl.localClippingEnabled = true
        // Calculate camera z so that the top and bottom are exactly at the edges of the fov
        // Based on https://stackoverflow.com/a/13351534/780425
        // Adding height for extra space to not clip horizontal lines exactly at 0/height in half.
        const maxLineWidth = 4
        const z = (height + maxLineWidth) / (2 * Math.tan((camera.fov / 360) * Math.PI))
        camera.position.set(width / 2, height / 2, z)
        return camera.lookAt(width / 2, height / 2, 0)
      }}
      dpr={window.devicePixelRatio}
      frameLoop="demand"
    >
      {children}
    </Canvas>
  )
}
