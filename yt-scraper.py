#!/usr/bin/env python3
"""
YouTube Metadata and Subtitle Scraper

This script scrapes metadata and subtitles from YouTube videos using yt-dlp.
It can extract video information, download subtitles in various formats, and
save the data in JSON format with timestamp information.

Requirements:
    pip install yt-dlp requests

Usage:
    python youtube_scraper.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import yt_dlp
import re


class YouTubeScraper:
    def __init__(self, output_dir: str = "scraped_data"):
        """
        Initialize the YouTube scraper.

        Args:
            output_dir: Directory to save scraped data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Configure yt-dlp options - focusing on Turkish and English only
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['tr', 'en', 'en-US', 'en-GB', 'tr-tr', 'en-orig'],  # Turkish and English variants only
            'skip_download': True,  # Only extract info, don't download video
        }

    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL.

        Args:
            url: YouTube URL

        Returns:
            Video ID or None if not found
        """
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def is_playlist_url(self, url: str) -> bool:
        """
        Check if the URL is a YouTube playlist.
        
        Args:
            url: YouTube URL
            
        Returns:
            True if URL is a playlist, False otherwise
        """
        playlist_patterns = [
            r'youtube\.com/playlist\?list=',
            r'youtube\.com/watch\?.*list=',
            r'youtu\.be/.*\?.*list=',
        ]
        
        for pattern in playlist_patterns:
            if re.search(pattern, url):
                return True
        return False

    def is_channel_url(self, url: str) -> bool:
        """
        Check if the URL is a YouTube channel.
        
        Args:
            url: YouTube URL
            
        Returns:
            True if URL is a channel, False otherwise
        """
        channel_patterns = [
            r'youtube\.com/channel/',
            r'youtube\.com/c/',
            r'youtube\.com/@',
            r'youtube\.com/user/',
        ]
        
        for pattern in channel_patterns:
            if re.search(pattern, url):
                return True
        return False

    def extract_videos_from_any_url(self, url: str) -> List[str]:
        """
        Extract video URLs from any YouTube URL (channel, playlist, or single video).
        
        Args:
            url: Any YouTube URL
            
        Returns:
            List of video URLs
        """
        video_urls = []
        
        try:
            # Configure yt-dlp to extract all videos from any URL type
            extract_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,  # Only extract URLs, don't download
                'ignoreerrors': True,
            }
            
            with yt_dlp.YoutubeDL(extract_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if info:
                    # Handle single video
                    if info.get('_type') != 'playlist' and info.get('id'):
                        video_url = f"https://www.youtube.com/watch?v={info['id']}"
                        video_urls.append(video_url)
                        print(f"Found single video: {info.get('title', 'Unknown')}")
                    
                    # Handle playlist/channel (multiple videos)
                    elif 'entries' in info:
                        valid_entries = [entry for entry in info['entries'] if entry and entry.get('id')]
                        print(f"Found {len(valid_entries)} videos")
                        
                        # Determine source type
                        source_type = "Unknown"
                        if info.get('_type') == 'playlist':
                            source_type = "Playlist"
                        elif self.is_channel_url(url):
                            source_type = "Channel"
                        
                        print(f"Source: {source_type} - {info.get('title', 'Unknown')}")
                        
                        for entry in valid_entries:
                            video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                            video_urls.append(video_url)
                            
        except Exception as e:
            print(f"Error extracting videos from URL: {str(e)}")
            
        return video_urls

    def extract_playlist_videos(self, playlist_url: str) -> List[str]:
        """
        Extract all video URLs from a YouTube playlist.
        
        Args:
            playlist_url: YouTube playlist URL
            
        Returns:
            List of video URLs
        """
        # Use the more general method
        return self.extract_videos_from_any_url(playlist_url)

    def parse_timestamp(self, timestamp_str: str) -> float:
        """
        Parse VTT timestamp to seconds.
        
        Args:
            timestamp_str: Timestamp string in format "HH:MM:SS.mmm"
            
        Returns:
            Time in seconds as float
        """
        try:
            # Handle format: HH:MM:SS.mmm or MM:SS.mmm
            parts = timestamp_str.split(':')
            if len(parts) == 3:  # HH:MM:SS.mmm
                hours, minutes, seconds = parts
                return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            elif len(parts) == 2:  # MM:SS.mmm
                minutes, seconds = parts
                return float(minutes) * 60 + float(seconds)
            else:
                return 0.0
        except (ValueError, IndexError):
            return 0.0

    def scrape_metadata(self, url: str) -> Dict[str, Any]:
        """
        Scrape metadata from a YouTube video.

        Args:
            url: YouTube video URL

        Returns:
            Dictionary containing video metadata
        """
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Filter subtitle languages to show only Turkish and English
                all_subtitle_langs = list(info.get('subtitles', {}).keys()) + list(info.get('automatic_captions', {}).keys())
                filtered_langs = [lang for lang in all_subtitle_langs if lang.startswith('tr') or lang.startswith('en')]

                # Extract relevant metadata
                metadata = {
                    'video_id': info.get('id'),
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'uploader': info.get('uploader'),
                    'uploader_id': info.get('uploader_id'),
                    'upload_date': info.get('upload_date'),
                    'duration': info.get('duration'),
                    'view_count': info.get('view_count'),
                    'like_count': info.get('like_count'),
                    'comment_count': info.get('comment_count'),
                    'thumbnail': info.get('thumbnail'),
                    'tags': info.get('tags', []),
                    'categories': info.get('categories', []),
                    'webpage_url': info.get('webpage_url'),
                    'channel_url': info.get('channel_url'),
                    'subscriber_count': info.get('channel_follower_count'),
                    'video_format': {
                        'width': info.get('width'),
                        'height': info.get('height'),
                        'fps': info.get('fps'),
                        'vcodec': info.get('vcodec'),
                        'acodec': info.get('acodec'),
                    },
                    'availability': info.get('availability'),
                    'age_limit': info.get('age_limit'),
                    'live_status': info.get('live_status'),
                    'available_subtitle_languages': filtered_langs  # Only Turkish and English
                }

                return metadata

        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            return {}

    def scrape_subtitles(self, url: str) -> Dict[str, Any]:
        """
        Scrape subtitles from a YouTube video with timestamp information.
        Focus on Turkish and English only.

        Args:
            url: YouTube video URL

        Returns:
            Dictionary containing subtitle data with timestamps
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            print("Could not extract video ID from URL")
            return {}

        subtitles = {}

        try:
            # Configure options for subtitle extraction - Turkish and English only
            subtitle_opts = {
                'quiet': True,
                'no_warnings': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['tr', 'en', 'en-US', 'en-GB', 'tr-tr', 'en-orig'],
                'skip_download': True,
                'outtmpl': str(self.output_dir / f'{video_id}.%(ext)s'),
            }

            with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Check for available subtitles
                available_subs = info.get('subtitles', {})
                auto_subs = info.get('automatic_captions', {})

                # Filter to only Turkish and English languages
                tr_en_manual_subs = {k: v for k, v in available_subs.items() if k.startswith('tr') or k.startswith('en')}
                tr_en_auto_subs = {k: v for k, v in auto_subs.items() if k.startswith('tr') or k.startswith('en')}

                print(f"Available manual subtitles (TR/EN): {list(tr_en_manual_subs.keys())}")
                print(f"Available auto subtitles (TR/EN): {list(tr_en_auto_subs.keys())}")

                # Process manual subtitles (Turkish and English only)
                for lang, subs in tr_en_manual_subs.items():
                    if subs:
                        print(f"Processing manual subtitle for language: {lang}")
                        # Try to get VTT format first, then others
                        for sub in subs:
                            if sub.get('ext') == 'vtt':
                                try:
                                    subtitle_content = ydl.urlopen(sub['url']).read().decode('utf-8')
                                    parsed_subs = self.parse_vtt_with_timestamps(subtitle_content)
                                    subtitles[f'{lang}_manual'] = parsed_subs
                                    print(f"Successfully extracted manual subtitle for {lang}")
                                    break
                                except Exception as e:
                                    print(f"Error downloading manual subtitle for {lang}: {e}")

                # Process automatic subtitles (Turkish and English only)
                # Prioritize specific languages
                priority_order = ['en', 'en-US', 'en-GB', 'en-orig', 'tr', 'tr-tr']
                processed_langs = set()

                for priority_lang in priority_order:
                    if priority_lang in tr_en_auto_subs and priority_lang not in processed_langs:
                        subs = tr_en_auto_subs[priority_lang]
                        if subs:
                            print(f"Processing auto subtitle for priority language: {priority_lang}")
                            for sub in subs:
                                if sub.get('ext') == 'vtt':
                                    try:
                                        subtitle_content = ydl.urlopen(sub['url']).read().decode('utf-8')
                                        parsed_subs = self.parse_vtt_with_timestamps(subtitle_content)
                                        subtitles[f'{priority_lang}_auto'] = parsed_subs
                                        processed_langs.add(priority_lang)
                                        print(f"Successfully extracted auto subtitle for {priority_lang}")
                                        break
                                    except Exception as e:
                                        print(f"Error downloading auto subtitle for {priority_lang}: {e}")

                # Process any remaining Turkish/English auto subtitles
                for lang, subs in tr_en_auto_subs.items():
                    if subs and lang not in processed_langs:
                        print(f"Processing remaining auto subtitle for language: {lang}")
                        for sub in subs:
                            if sub.get('ext') == 'vtt':
                                try:
                                    subtitle_content = ydl.urlopen(sub['url']).read().decode('utf-8')
                                    parsed_subs = self.parse_vtt_with_timestamps(subtitle_content)
                                    subtitles[f'{lang}_auto'] = parsed_subs
                                    print(f"Successfully extracted auto subtitle for {lang}")
                                    break
                                except Exception as e:
                                    print(f"Error downloading auto subtitle for {lang}: {e}")
                                    break

        except Exception as e:
            print(f"Error extracting subtitles: {str(e)}")

        return subtitles

    def parse_vtt_with_timestamps(self, vtt_content: str) -> Dict[str, Any]:
        """
        Parse VTT subtitle content with timestamp information.

        Args:
            vtt_content: Raw VTT subtitle content

        Returns:
            Dictionary containing subtitles with timestamps and cleaned text
        """
        lines = vtt_content.split('\n')
        subtitles_with_timestamps = []
        cleaned_text_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for timestamp lines (format: start --> end)
            timestamp_match = re.match(r'(\d{1,2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}\.\d{3})', line)
            if timestamp_match:
                start_time = timestamp_match.group(1)
                end_time = timestamp_match.group(2)
                
                # Parse timestamps to seconds
                start_seconds = self.parse_timestamp(start_time)
                end_seconds = self.parse_timestamp(end_time)
                
                # Get the subtitle text (next non-empty lines)
                subtitle_text = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    text_line = lines[i].strip()
                    # Remove HTML tags and styling
                    cleaned_line = re.sub(r'<[^>]+>', '', text_line)
                    if cleaned_line:
                        subtitle_text.append(cleaned_line)
                        cleaned_text_lines.append(cleaned_line)
                    i += 1
                
                if subtitle_text:
                    subtitles_with_timestamps.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_seconds': start_seconds,
                        'end_seconds': end_seconds,
                        'duration_seconds': round(end_seconds - start_seconds, 3),
                        'text': ' '.join(subtitle_text)
                    })
            else:
                i += 1
        
        return {
            'timed_subtitles': subtitles_with_timestamps,
            'full_text': '\n'.join(cleaned_text_lines),
            'total_segments': len(subtitles_with_timestamps),
            'total_duration': max([seg['end_seconds'] for seg in subtitles_with_timestamps]) if subtitles_with_timestamps else 0
        }

    def clean_vtt_content(self, vtt_content: str) -> str:
        """
        Clean VTT subtitle content to extract only the text.

        Args:
            vtt_content: Raw VTT subtitle content

        Returns:
            Cleaned subtitle text
        """
        lines = vtt_content.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip VTT headers, timestamps, and empty lines
            if (line and
                not line.startswith('WEBVTT') and
                not re.match(r'\d{1,2}:\d{2}:\d{2}\.\d{3}', line) and
                not line.startswith('NOTE') and
                not re.match(r'^\d+$', line)):

                # Remove HTML tags and styling
                cleaned_line = re.sub(r'<[^>]+>', '', line)
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)

        return '\n'.join(cleaned_lines)

    def scrape_video(self, url: str) -> Dict[str, Any]:
        """
        Scrape both metadata and subtitles from a YouTube video.

        Args:
            url: YouTube video URL

        Returns:
            Dictionary containing all scraped data
        """
        print(f"Scraping video: {url}")

        # Extract metadata
        metadata = self.scrape_metadata(url)
        if not metadata:
            print("Failed to extract metadata")
            return {}

        # Extract subtitles
        subtitles = self.scrape_subtitles(url)

        # Combine data
        scraped_data = {
            'url': url,
            'metadata': metadata,
            'subtitles': subtitles,
            'scraped_at': self.get_current_timestamp()
        }

        return scraped_data

    def scrape_multiple_videos(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple YouTube videos.

        Args:
            urls: List of YouTube video URLs

        Returns:
            List of dictionaries containing scraped data
        """
        results = []

        for i, url in enumerate(urls, 1):
            print(f"\nProcessing video {i}/{len(urls)}")
            print("-" * 40)
            data = self.scrape_video(url)
            if data:
                results.append(data)
                print(f"✓ Successfully scraped video {i}")
            else:
                print(f"✗ Failed to scrape video {i}")

        return results

    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename by removing invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for filesystem
        """
        # Remove invalid characters for filenames
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '')
        
        # Replace multiple spaces with single space
        filename = re.sub(r'\s+', ' ', filename)
        
        # Trim whitespace and limit length
        filename = filename.strip()[:100]  # Limit to 100 characters
        
        # If filename becomes empty, use fallback
        if not filename:
            filename = "untitled_video"
            
        return filename

    def save_data(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        Save scraped data to JSON file.

        Args:
            data: Scraped data dictionary
            filename: Optional filename (will auto-generate if not provided)

        Returns:
            Path to saved file
        """
        if not filename:
            # Use video title as filename if available
            title = data.get('metadata', {}).get('title')
            if title:
                sanitized_title = self.sanitize_filename(title)
                filename = f"{sanitized_title}.json"
            else:
                video_id = data.get('metadata', {}).get('video_id', 'unknown')
                filename = f"{video_id}_scraped.json"

        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Data saved to: {filepath}")
        return str(filepath)

    def save_multiple_data(self, data_list: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Save multiple scraped videos data to JSON file and individual files.
        
        Args:
            data_list: List of scraped data dictionaries
            filename: Optional filename for combined data (will auto-generate if not provided)
            
        Returns:
            Path to saved combined file
        """
        # Save individual video files
        individual_files = []
        print(f"\nSaving individual video files...")
        
        for i, video_data in enumerate(data_list, 1):
            try:
                # Generate filename from video title
                title = video_data.get('metadata', {}).get('title')
                if title:
                    sanitized_title = self.sanitize_filename(title)
                    individual_filename = f"{sanitized_title}.json"
                else:
                    video_id = video_data.get('metadata', {}).get('video_id', f'video_{i}')
                    individual_filename = f"{video_id}.json"
                
                # Save individual file
                individual_path = self.save_data(video_data, individual_filename)
                individual_files.append(individual_path)
                print(f"  ✓ Saved: {individual_filename}")
                
            except Exception as e:
                print(f"  ✗ Failed to save video {i}: {str(e)}")
        
        # Save combined playlist file
        if not filename:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"playlist_combined_{timestamp}.json"
            
        filepath = self.output_dir / filename
        
        # Create summary data
        summary_data = {
            'scraped_at': self.get_current_timestamp(),
            'total_videos': len(data_list),
            'individual_files': [os.path.basename(f) for f in individual_files],
            'videos': data_list
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
        print(f"\nCombined playlist data saved to: {filepath}")
        print(f"Individual video files: {len(individual_files)}")
        return str(filepath)

def main():
    """Main function to demonstrate the scraper."""
    scraper = YouTubeScraper()

    # Example usage
    print("YouTube Metadata and Subtitle Scraper")
    print("=" * 40)
    print("Supports any YouTube link:")
    print("• Individual videos")
    print("• Playlists") 
    print("• Channels")
    print("• User pages")
    print("• Any other YouTube URL")

    # Get video URL from user input
    url = input("\nEnter any YouTube URL: ").strip()

    if not url:
        print("No URL provided. Using example URL...")
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example URL

    # Extract videos from any URL type
    try:
        print(f"\n🔍 Analyzing URL...")
        video_urls = scraper.extract_videos_from_any_url(url)
        
        if not video_urls:
            print("❌ No videos found at the provided URL")
            return
        
        # Determine if it's a single video or multiple videos
        if len(video_urls) == 1:
            print(f"\n🎬 Found single video")
            data = scraper.scrape_video(video_urls[0])

            if data:
                # Display some basic info
                metadata = data.get('metadata', {})
                title = metadata.get('title', 'N/A')
                print(f"\nTitle: {title}")
                print(f"Uploader: {metadata.get('uploader', 'N/A')}")
                print(f"Duration: {metadata.get('duration', 'N/A')} seconds")
                print(f"View Count: {metadata.get('view_count', 'N/A')}")
                print(f"Available subtitle languages (TR/EN only): {metadata.get('available_subtitle_languages', [])}")
                
                # Display subtitle information with timing
                subtitles = data.get('subtitles', {})
                print(f"\nScraped Subtitles:")
                for subtitle_key, subtitle_data in subtitles.items():
                    print(f"  - {subtitle_key}:")
                    print(f"    Total segments: {subtitle_data.get('total_segments', 0)}")
                    print(f"    Total duration: {subtitle_data.get('total_duration', 0):.1f} seconds")
                    
                    # Show first few subtitle segments as example
                    timed_subs = subtitle_data.get('timed_subtitles', [])
                    if timed_subs:
                        print(f"    First 3 segments:")
                        for i, segment in enumerate(timed_subs[:3]):
                            print(f"      {segment['start_time']} -> {segment['end_time']}: {segment['text'][:50]}...")

                # Save data (filename will be based on video title)
                filepath = scraper.save_data(data)
                sanitized_title = scraper.sanitize_filename(title) if title != 'N/A' else 'unknown'
                print(f"Video saved as: {sanitized_title}.json")
                
                # Example: Find subtitle at specific time
                if subtitles:
                    test_time = 30.0  # 30 seconds
                    # Try to find English subtitles first, then any available
                    lang_keys = [k for k in subtitles.keys() if 'en' in k]
                    if not lang_keys:
                        lang_keys = list(subtitles.keys())
                    
                    if lang_keys:
                        first_lang = lang_keys[0]
                        segment = scraper.find_subtitle_at_time(subtitles, test_time, first_lang)
                        if segment:
                            print(f"\nSubtitle at {test_time} seconds ({first_lang}):")
                            print(f"  Time: {segment['start_time']} -> {segment['end_time']}")
                            print(f"  Text: {segment['text']}")

            else:
                print("Failed to scrape video data")
                
        else:
            # Multiple videos found
            print(f"\n📹 Found {len(video_urls)} videos")
            
            # Ask user if they want to proceed
            proceed = input(f"\nProceed to scrape all {len(video_urls)} videos? (y/n): ").lower().strip()
            if proceed != 'y':
                print("Operation cancelled")
                return
                
            print(f"\n🚀 Starting to scrape {len(video_urls)} videos...")
            all_data = scraper.scrape_multiple_videos(video_urls)
            
            if all_data:
                print(f"\n✅ Successfully scraped {len(all_data)} out of {len(video_urls)} videos")
                
                # Save all data (both individual and combined)
                filepath = scraper.save_multiple_data(all_data)
                
                # Display summary
                print(f"\n📊 Summary:")
                print(f"Total videos processed: {len(video_urls)}")
                print(f"Successfully scraped: {len(all_data)}")
                print(f"Failed: {len(video_urls) - len(all_data)}")
                print(f"Each video saved as individual JSON file named after video title")
                
                # Show some sample data
                for i, data in enumerate(all_data[:3]):  # Show first 3 videos
                    metadata = data.get('metadata', {})
                    title = metadata.get('title', 'N/A')
                    sanitized_title = scraper.sanitize_filename(title) if title != 'N/A' else 'N/A'
                    print(f"\nVideo {i+1}:")
                    print(f"  Title: {title}")
                    print(f"  Filename: {sanitized_title}.json")
                    print(f"  Uploader: {metadata.get('uploader', 'N/A')}")
                    print(f"  Duration: {metadata.get('duration', 'N/A')} seconds")
                    
                if len(all_data) > 3:
                    print(f"\n... and {len(all_data) - 3} more videos")
                    
            else:
                print("Failed to scrape any videos")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()  