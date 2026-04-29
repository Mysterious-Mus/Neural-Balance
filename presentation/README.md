# Presentation

This folder contains a brief Marp presentation for the project report.

## Files

- `slides.md`: main Marp slide deck
- `speaker-script.md`: exact script for the speaker, slide by slide

The deck is designed to be short (`11` slides). The spoken script is embedded in `slides.md` as Marp presenter notes so it can be exported into a normal PowerPoint file, and it is also kept in `speaker-script.md` as a separate rehearsal copy.

## Preview

If you use the Marp extension in VS Code or Cursor, open `slides.md` and start the Marp preview.

## Export from the command line

On our server, npx is ok, but need to rootlessly install chrome first

```
cd ~
mkdir -p chrome-install
cd chrome-install
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
ar x google-chrome-stable_current_amd64.deb
tar xf data.tar.xz
# append path to bashrc
echo 'export PATH=$PATH:~/chrome-install/opt/google/chrome' >> ~/.bashrc
# also chrome no sandbox
echo 'export CHROME_NO_SANDBOX=1' >> ~/.bashrc
```

Then, you can export with:

```bash
# refresh .bashrc
source ~/.bashrc
cd PATH_TO_REPO/presentation/
npx @marp-team/marp-cli slides.md --html
npx @marp-team/marp-cli slides.md --pdf
npx @marp-team/marp-cli slides.md --pptx
```

Run those commands from the `presentation/` folder.

For PowerPoint with speaker notes, use the normal `--pptx` export. Do not use `--pptx-editable`, because editable PPTX does not preserve presenter notes.

## Notes

- The content is based on the report in `REPORT/`
- Numeric results are taken from the current report values and evaluation table
- If the report changes, update the results slide accordingly
