#!/usr/bin/env python3

import argparse
import sys
from typing import List
from src.youtube_downloader import YouTubeDownloader

def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download videos from YouTube and optionally convert to MP3.'
    )
    
    parser.add_argument(
        'urls',
        nargs='+',
        help='One or more YouTube URLs to download'
    )
    
    parser.add_argument(
        '-a', '--audio',
        action='store_true',
        help='Convert to MP3 after download'
    )
    
    parser.add_argument(
        '-d', '--delete',
        action='store_true',
        help='Delete video file after converting to MP3'
    )
    
    parser.add_argument(
        '--ffmpeg',
        help='Path to ffmpeg executable'
    )
    
    return parser.parse_args(args)

def main(args: List[str] = None) -> int:
    """Main entry point for the application."""
    if args is None:
        args = sys.argv[1:]
    
    parsed_args = parse_args(args)
    
    try:
        downloader = YouTubeDownloader(ffmpeg_path=parsed_args.ffmpeg)
        
        for url in parsed_args.urls:
            print(f"\nProcessing: {url}")
            downloader.process_url(
                url,
                convert_to_audio=parsed_args.audio,
                delete_video=parsed_args.delete
            )
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
