// swift-tools-version: 5.8
import PackageDescription

let package = Package(
    name: "stable-diffusion",
    platforms: [
        .macOS(.v11),
        .iOS(.v14),
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
