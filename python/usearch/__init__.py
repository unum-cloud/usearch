import os
import platform
import tempfile
import warnings
import urllib.request
from typing import Optional
from urllib.error import HTTPError


from usearch.compiled import (
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_PATCH,
)

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"


class BinaryManager:
    def __init__(self, version: Optional[str] = None):
        if version is None:
            version = __version__
        self.version = version or __version__
        self.download_dir = self.determine_download_dir()

    @staticmethod
    def determine_download_dir():
        # Check if running within a virtual environment
        virtual_env = os.getenv("VIRTUAL_ENV")
        if virtual_env:
            # Use a subdirectory within the virtual environment for binaries
            return os.path.join(virtual_env, "bin", "usearch_binaries")
        else:
            # Fallback to a directory in the user's home folder
            home_dir = os.path.expanduser("~")
            return os.path.join(home_dir, ".usearch", "binaries")

    def sqlite_file_name(self) -> str:
        version = self.version
        os_map = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}
        arch_map = {
            "x86_64": "amd64" if platform.system() != "Darwin" else "x86_64",
            "AMD64": "amd64",
            "arm64": "arm64",
            "aarch64": "arm64",
            "x86": "x86",
        }
        os_part = os_map.get(platform.system(), "")
        arch = platform.machine()
        arch_part = arch_map.get(arch, "")
        extension = {"Linux": "so", "Windows": "dll", "Darwin": "dylib"}.get(platform.system(), "")
        filename = f"usearch_sqlite_{os_part}_{arch_part}_{version}.{extension}"
        return filename

    def sqlite_download_url(self) -> str:
        version = self.version
        filename = self.sqlite_file_name()
        base_url = "https://github.com/unum-cloud/usearch/releases/download"
        url = f"{base_url}/v{version}/{filename}"
        return url

    def download_binary(self, url: str, dest_folder: str) -> str:
        """
        Downloads a file from a given URL to a specified destination folder.

        Args:
            url (str): The URL to download the file from.
            dest_folder (str): The folder where the file will be saved.

        Returns:
            The path to the downloaded file.
        """
        filename = url.split("/")[-1]
        dest_path = os.path.join(dest_folder, filename)
        urllib.request.urlretrieve(url, dest_path)
        return dest_path

    def sqlite_found_or_downloaded(self) -> Optional[str]:
        """
        Attempts to locate the pre-installed `usearch_sqlite` binary.
        If not found, downloads it from GitHub.

        Returns:
            The path to the binary if found or downloaded, otherwise None.
        """
        # Search local directories
        local_dirs = ["build", "build_artifacts", "build_release", "build_debug"]
        extensions = {"Linux": ".so", "Windows": ".dll", "Darwin": ".dylib"}
        os_type = platform.system()
        file_extension = extensions.get(os_type, "")

        # Check local development directories first
        for directory in local_dirs:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(file_extension) and "usearch_sqlite" in file:
                        return os.path.join(root, file).removesuffix(file_extension)

        # Check a temporary directory (assuming the binary might be downloaded from a GitHub release)
        local_path = os.path.join(self.download_dir, self.sqlite_file_name())
        if os.path.exists(local_path):
            return local_path.removesuffix(file_extension)

        # If not found locally, warn the user and download from GitHub
        warnings.warn("Will download `usearch_sqlite` binary from GitHub.", UserWarning)
        try:
            binary_path = self.download_binary(self.sqlite_download_url(), self.download_dir)
        except HTTPError as e:
            # If the download fails due to HTTPError (e.g., 404 Not Found), like a missing lib version
            if e.code == 404:
                warnings.warn(f"Download failed: {e.url} could not be found.", UserWarning)
            else:
                warnings.warn(f"Download failed with HTTP error: {e.code} {e.reason}", UserWarning)
            return None

        # Handle the case where binary_path does not exist after supposed successful download
        if os.path.exists(binary_path):
            return binary_path.removesuffix(file_extension)
        else:
            warnings.warn("Failed to download `usearch_sqlite` binary from GitHub.", UserWarning)
            return None


def sqlite_path(version: str = None) -> str:
    manager = BinaryManager(version=version)
    result = manager.sqlite_found_or_downloaded()
    if result is None:
        raise FileNotFoundError("Failed to find or download `usearch_sqlite` binary.")
    return result
