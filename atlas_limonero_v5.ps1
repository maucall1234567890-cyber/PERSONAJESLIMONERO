param([string]$Carpeta = 'Limonero_v5')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$refs = @([Drawing.Bitmap].Assembly.Location,[Drawing.Rectangle].Assembly.Location)
if ($PSVersionTable.PSEdition -eq 'Core') {
    $refs += @('System.Private.Windows.Core','System.Private.Windows.GdiPlus','System.ComponentModel.Primitives','System.ComponentModel.TypeConverter','System.Runtime.InteropServices')
}
Add-Type -ReferencedAssemblies ($refs | Select-Object -Unique) -TypeDefinition @'
using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;
using System.Runtime.InteropServices;
public static class AtlasLimonero {
    static Rectangle Bounds(Bitmap bmp) {
        var all = new Rectangle(0,0,bmp.Width,bmp.Height);
        var data = bmp.LockBits(all, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        int x1=bmp.Width, y1=bmp.Height, x2=-1, y2=-1;
        var row = new byte[bmp.Width*4];
        try {
            for(int y=0;y<bmp.Height;y++) {
                Marshal.Copy(IntPtr.Add(data.Scan0,y*data.Stride),row,0,row.Length);
                for(int x=0;x<bmp.Width;x++) {
                    if(row[4*x+3]==0) continue;
                    x1=Math.Min(x,x1); x2=Math.Max(x,x2);
                    y1=Math.Min(y,y1); y2=Math.Max(y,y2);
                }
            }
        } finally { bmp.UnlockBits(data); }
        if(x2<x1 || x1==0 || y1==0 || x2==bmp.Width-1 || y2==bmp.Height-1)
            throw new Exception("El sprite está vacío, es opaco o toca un borde.");
        return Rectangle.FromLTRB(x1,y1,x2+1,y2+1);
    }
    public static string Build(string folder) {
        Rectangle bounds=Rectangle.Empty;
        const int count=72, columns=9;
        for(int i=1;i<=count;i++) {
            using(var bmp=new Bitmap(Path.Combine(folder,"sprites","Reposo_"+i.ToString("D4")+".png"))) {
                if(bmp.Width!=512 || bmp.Height!=512) throw new Exception("Tamaño de cuadro incorrecto.");
                var rect=Bounds(bmp);
                bounds=i==1?rect:Rectangle.Union(bounds,rect);
            }
        }
        int w=((bounds.Width+8+15)/16)*16, h=((bounds.Height+8+15)/16)*16;
        var crop=new Rectangle(bounds.Left-(w-bounds.Width)/2,bounds.Top-(h-bounds.Height)/2,w,h);
        string path=Path.Combine(folder,"Limonero_reposo_atlas.png");
        using(var sheet=new Bitmap(columns*w,8*h,PixelFormat.Format32bppArgb))
        using(var g=Graphics.FromImage(sheet)) {
            g.CompositingMode=CompositingMode.SourceCopy;
            for(int i=0;i<count;i++) {
                using(var bmp=new Bitmap(Path.Combine(folder,"sprites","Reposo_"+(i+1).ToString("D4")+".png")))
                using(var tile=bmp.Clone(crop,PixelFormat.Format32bppArgb)) {
                    g.DrawImageUnscaled(tile,(i%columns)*w,(i/columns)*h);
                }
            }
            sheet.Save(path,ImageFormat.Png);
        }
        using(var check=new Bitmap(path)) {
            if(check.Width!=columns*w || check.Height!=8*h || check.GetPixel(0,0).A!=0)
                throw new Exception("Error verificando el atlas.");
        }
        return String.Format("72 frames; 9 x 8; tile {0} x {1}; crop {2},{3}; baseline {4}; center X {5}",
                             w,h,crop.X,crop.Y,bounds.Bottom-crop.Y,256-crop.X);
    }
}
'@
[AtlasLimonero]::Build((Join-Path $PSScriptRoot $Carpeta))
