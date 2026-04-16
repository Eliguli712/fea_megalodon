import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.File;
import java.io.IOException;

public class ExportStrict3BdfStd1Preview {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";
  private static final String PNG =
      "DICOM_16um/exports/std1_sol6_preview.png";

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    try {
      model.result().remove("pg_std1_preview");
    } catch (Exception ignored) {
    }
    model.result().create("pg_std1_preview", "PlotGroup3D");
    model.result("pg_std1_preview").label("std1/sol6 preview");
    model.result("pg_std1_preview").set("data", "dset6");
    model.result("pg_std1_preview").create("surf1", "Surface");
    ResultFeature surf = model.result("pg_std1_preview").feature("surf1");
    surf.set("expr", new String[] {"solid.mises"});
    try {
      surf.set("descr", new String[] {"Von Mises stress"});
    } catch (Exception ignored) {
    }
    try {
      surf.set("colortable", "RainbowLight");
    } catch (Exception ignored) {
    }
    try {
      surf.set("resolution", "coarse");
    } catch (Exception ignored) {
    }

    model.result("pg_std1_preview").run();

    try {
      model.result().export().remove("img_std1_preview");
    } catch (Exception ignored) {
    }
    model.result().export().create("img_std1_preview", "Image3D");
    model.result().export("img_std1_preview").set("plotgroup", "pg_std1_preview");
    try {
      model.result().export("img_std1_preview").set("imagetype", "png");
    } catch (Exception ignored) {
    }
    try {
      model.result().export("img_std1_preview").set("unit", "px");
      model.result().export("img_std1_preview").set("width", 1200);
      model.result().export("img_std1_preview").set("height", 900);
      model.result().export("img_std1_preview").set("lockratio", "off");
      model.result().export("img_std1_preview").set("zoomextents", "on");
    } catch (Exception ignored) {
    }

    File out = new File(PNG);
    File parent = out.getParentFile();
    if (parent != null && !parent.exists()) {
      parent.mkdirs();
    }

    model.result().export("img_std1_preview").set("pngfilename", PNG);
    model.result().export("img_std1_preview").run();

    long size = out.exists() ? out.length() : 0L;
    System.out.println("STD1_PREVIEW_EXPORT|file=" + PNG + "|exists=" + out.exists() + "|size=" + size);
  }
}
