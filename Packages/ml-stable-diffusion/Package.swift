// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "stable-diffusion",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(name: "StableDiffusion", targets: ["StableDiffusion"]),
    ],
    targets: [
        .target(
            name: "StableDiffusion",
            path: "Sources/StableDiffusion"
        ),
    ]
)
