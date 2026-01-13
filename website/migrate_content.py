#!/usr/bin/env python3
"""Migrate content from Astro/Starlight to Nextra format."""

import os
import re
from pathlib import Path

# Mapping from Astro paths to Nextra paths
PATH_MAPPING = {
    "src/content/docs/getting-started/": "pages/getting-started/",
    "src/content/docs/guides/": "pages/guides/",
    "src/content/docs/concepts/": "pages/concepts/",
    "src/content/docs/reference/": "pages/reference/",
    "src/content/docs/index.mdx": "pages/index.mdx",
}


def convert_frontmatter(content: str) -> str:
    """Convert Astro frontmatter to Nextra format."""
    if not content.startswith("---"):
        return content

    # Extract frontmatter
    parts = content.split("---", 2)
    if len(parts) < 3:
        return content

    frontmatter = parts[1].strip()
    body = parts[2]

    # Parse frontmatter
    title = None
    description = None
    order = None

    for line in frontmatter.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("title:"):
            title = line.split("title:", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("description:"):
            description = line.split("description:", 1)[1].strip().strip('"').strip("'")
        elif "order:" in line:
            order = line.split("order:", 1)[1].strip()

    # Create Nextra frontmatter (simpler format)
    nextra_fm = []
    if title:
        nextra_fm.append(f"title: {title}")
    if description:
        nextra_fm.append(f"description: {description}")

    if nextra_fm:
        return "---\n" + "\n".join(nextra_fm) + "\n---\n\n" + body
    else:
        return body


def remove_astro_components(content: str) -> str:
    """Remove Astro-specific components and imports."""
    # Remove CardGrid and Card components
    content = re.sub(
        r'<CardGrid[^>]*>.*?</CardGrid>',
        '',
        content,
        flags=re.DOTALL
    )
    content = re.sub(
        r'<Card[^>]*>.*?</Card>',
        '',
        content,
        flags=re.DOTALL
    )
    # Remove Astro imports
    content = re.sub(
        r'import\s+\{[^}]*\}\s+from\s+[\'"]@astrojs/starlight[^"\']*["\'];?\s*\n',
        '',
        content
    )
    return content


def update_links(content: str) -> str:
    """Update internal links to work with Nextra routing."""
    # Links already have /loclean/ prefix, keep them
    # Just ensure they're correct
    return content


def migrate_file(src_path: Path, dst_path: Path) -> None:
    """Migrate a single file from Astro to Nextra format."""
    print(f"Migrating: {src_path} -> {dst_path}")

    # Read source file
    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Convert
    content = convert_frontmatter(content)
    content = remove_astro_components(content)
    content = update_links(content)

    # Ensure directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Write destination file
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    """Main migration function."""
    base_dir = Path(__file__).parent

    # Find all content files
    src_dir = base_dir / "src" / "content" / "docs"
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        return

    # Migrate files
    for src_file in src_dir.rglob("*.md*"):
        # Get relative path
        rel_path = src_file.relative_to(base_dir / "src" / "content" / "docs")

        # Determine destination
        if str(rel_path) == "index.mdx":
            dst_path = base_dir / "pages" / "index.mdx"
        else:
            # Remove extension, add .mdx
            dst_path = base_dir / "pages" / rel_path.with_suffix(".mdx")

        migrate_file(src_file, dst_path)

    print("\nMigration complete!")


if __name__ == "__main__":
    main()
