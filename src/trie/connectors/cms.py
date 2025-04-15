from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import zipfile

from bs4 import BeautifulSoup
import requests
from rich.console import Console

BASE_URL = "https://www.cms.gov/medicare/coding-billing/icd-10-codes"
CURRENT_YEAR = datetime.now().year
MONTH_PATTERN = re.compile(
    r"\b(december|november|october|september|august|july|june|may|april|march|february|january)\b", re.IGNORECASE
)


def extract_zip_file(path_to_zip: Path, download_path: Path) -> None:
    with zipfile.ZipFile(path_to_zip) as zip_file:
        for member in zip_file.namelist():
            filename = os.path.basename(member)
            if not filename:
                continue
            # copy file (taken from zipfile's extract)
            source = zip_file.open(member)
            target = open(os.path.join(download_path, filename), "wb")
            with source, target:
                shutil.copyfileobj(source, target)


def maybe_get_updated_files(files: list[Path], use_update: bool) -> list[Path]:
    """Get the updated version of the files."""
    # Filter for updated (month-tagged) files if requested
    pdfs = [link for link in files if link.name.endswith(".pdf")]
    zips = [link for link in files if link.name.endswith(".zip")]
    # if use_update:
    #     # Prefer only updated (month-tagged) files if available
    #     month_links = [link for link in zips if MONTH_PATTERN.search(link.name) and link.name.endswith(".zip")]
    #     if month_links:
    #         download_links = month_links + pdfs
    #         with Console() as console:
    #             console.print("[bold yellow]Using updated ICD version for year.[/bold yellow]")
    #     else:
    #         download_links = zips
    #         with Console() as console:
    #             console.print("[bold yellow]No updated version found, using base ICD files for.[/bold yellow]")
    # else:
    #     # Explicitly exclude any month-tagged files
    #     download_links = [link for link in zips if not MONTH_PATTERN.search(link.name)] + pdfs
    return pdfs + zips


def download_cms_icd_version(download_path: Path, year: int, use_update: bool) -> None:
    """Download the ICD version for the specified year from the CMS website."""
    response = requests.get(BASE_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    all_links = [
        Path(link["href"]) for link in soup.find_all("a", href=True) if link["href"].endswith((".zip", ".pdf"))
    ]
    version_links = [link for link in all_links if str(year) in link.as_posix()]

    if not version_links:
        available_years = sorted(
            set(
                year
                for link in all_links
                for year in re.findall(r"20\d{2}", str(link.name))
                if int(year) <= CURRENT_YEAR and int(year) >= 2015
            )
        )
        raise ValueError(f"No files found for year {year}. Available years: {available_years}")

    download_links = maybe_get_updated_files(version_links, use_update=use_update)
    downloaded_file_names = [f.name for f in download_path.glob("*")]
    new_download_links = [link for link in download_links if Path(link.name).name not in downloaded_file_names]

    for file_path in new_download_links:
        file_url = f"https://www.cms.gov{file_path.as_posix()}"
        file_name = download_path / Path(file_url).name

        with Console() as console:
            console.print(f"[bold blue]Downloading ICD version {Path(file_url).name}...[/bold blue]")

        file_response = requests.get(file_url)
        file_response.raise_for_status()

        with open(file_name, "wb") as file:
            file.write(file_response.content)

        if file_name.suffix == ".zip":
            extract_zip_file(file_name, download_path)

    with Console() as console:
        console.print(f"[bold green]Downloaded ICD version {year} successfully![/bold green]")
