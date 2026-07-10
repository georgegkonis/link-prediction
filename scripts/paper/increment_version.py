import os
from pathlib import Path

def main():
    paper_dir = Path("paper")
    version_file = paper_dir / ".version"
    tex_file = paper_dir / "version.tex"
    
    version = 0
    if version_file.exists():
        try:
            version = int(version_file.read_text().strip())
        except ValueError:
            pass
            
    version += 1
    version_file.write_text(str(version) + "\n")
    
    tex_content = f"\\newcommand{{\\draftversion}}{{DRAFT v{version}}}\n"
    tex_file.write_text(tex_content)
    
    print(f"Incremented paper draft version to {version}")

if __name__ == "__main__":
    main()
