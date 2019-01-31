# Partti

Partti is a program which reads sheet music images and plays the sound represented in the sheet.

## Usage

Execute the main script with a `-i` argument where you specify the path to the image. Example:

```bash
python3 main.py -i ./data/ode_to_joy.jpg
```

## Limitations

The input images work better if they are clean, good illumination and no reflections, a clear distinction between the background and the white paper.

The software still cannot detect tempo correctly, no alterations, it does not detect time signatures nor armatures, and it can detect only one voice in the staff (only one note vertically at a time), if there's more than one, it will only play the topmost one.
