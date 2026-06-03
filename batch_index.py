"""
Batch indexer: indexes all images from a folder into the Vision MCP repository.

Usage:
    python batch_index.py <folder_path> [--api-url http://localhost:8001] [--tags tag1,tag2]

Example:
    python batch_index.py ./imagenes
    python batch_index.py ./imagenes --api-url http://api:8001 --tags naturaleza,paisaje
"""
import argparse
import sys
import time
from pathlib import Path

import requests

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(folder: Path) -> list[Path]:
    """Recursively collect all supported image files from a folder."""
    images = []
    for f in sorted(folder.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(f)
    return images


def index_image(api_url: str, image_path: Path, tags: str, timeout: int = 600) -> dict:
    """Send a single image to the API for indexing."""
    suffix = image_path.suffix.lower().lstrip(".")
    mime = f"image/{'jpeg' if suffix == 'jpg' else suffix}"

    with open(image_path, "rb") as fh:
        resp = requests.post(
            f"{api_url}/repository/add",
            files={"file": (image_path.name, fh, mime)},
            data={"tags": tags},
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Batch index images into Vision MCP repository")
    parser.add_argument("folder", type=str, help="Path to folder containing images")
    parser.add_argument("--api-url", type=str, default="http://localhost:8001",
                        help="API base URL (default: http://localhost:8001)")
    parser.add_argument("--tags", type=str, default="",
                        help="Comma-separated tags applied to all images")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout per image in seconds (default: 600)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)

    # Check API health
    try:
        health = requests.get(f"{args.api_url}/health", timeout=5)
        health.raise_for_status()
        info = health.json()
        print(f"✅ API connected | Ollama: {info.get('ollama_available')} | "
              f"Model: {info.get('ollama_model')} | "
              f"Repo images: {info.get('repository_images', 0)}")
    except Exception as e:
        print(f"❌ Cannot reach API at {args.api_url}: {e}")
        sys.exit(1)

    images = collect_images(folder)
    total = len(images)

    if total == 0:
        print(f"⚠️  No supported images found in {folder}")
        print(f"   Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(0)

    print(f"\n📁 Found {total} image(s) in {folder}")
    print(f"🏷️  Tags: {args.tags or '(none)'}")
    print(f"⏱️  Timeout per image: {args.timeout}s")
    print("-" * 60)

    success = 0
    failed = 0
    start_time = time.time()

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{total}] {img_path.name} ... ", end="", flush=True)
        try:
            result = index_image(args.api_url, img_path, args.tags, args.timeout)
            img_id = result.get("image_id", "?")
            dims = result.get("dimensions", "?")
            print(f"✅ {img_id[:8]}... | {dims}")
            success += 1
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection refused — is the API running?")
            failed += total - i + 1
            break
        except requests.exceptions.Timeout:
            print(f"❌ Timeout ({args.timeout}s) — model too slow on CPU")
            failed += 1
        except requests.exceptions.HTTPError as e:
            detail = e.response.text[:200] if e.response is not None else str(e)
            print(f"❌ HTTP {e.response.status_code if e.response else '?'}: {detail}")
            failed += 1
        except Exception as e:
            print(f"❌ {e}")
            failed += 1

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"🏁 Done in {elapsed:.0f}s | ✅ {success} indexed | ❌ {failed} failed | "
          f"📊 {total} total")


if __name__ == "__main__":
    main()
