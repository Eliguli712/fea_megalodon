import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;

public class RunStrict3BdfHighResVonMisesExports {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution.mph";
  private static final String OUT_DIR =
      "DICOM_16um/exports/highres_von_mises";

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    String[] exportTags = model.result().export().tags();
    int found = 0;
    int ok = 0;
    int fail = 0;

    if (exportTags != null) {
      for (String tag : exportTags) {
        if (tag == null || !tag.startsWith("img_hr_")) {
          continue;
        }
        found++;

        String file = "";
        try {
          file = model.result().export(tag).getString("pngfilename");
        } catch (Exception ignored) {
        }

        try {
          model.result().export(tag).run();
          ok++;
          System.out.println("EXPORT_RUN|tag=" + tag + "|ok=true|file=" + file);
        } catch (Exception ex) {
          fail++;
          System.out.println(
              "EXPORT_RUN|tag=" + tag + "|ok=false|file=" + file + "|err=" + ex.getMessage());
        }
      }
    }

    System.out.println("EXPORT_FOUND|" + found);
    System.out.println("EXPORT_OK|" + ok);
    System.out.println("EXPORT_FAIL|" + fail);
    System.out.println("OUT_DIR|" + OUT_DIR);
  }
}
