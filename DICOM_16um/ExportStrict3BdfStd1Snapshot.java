import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.File;
import java.io.IOException;

public class ExportStrict3BdfStd1Snapshot {
  private static final String MPH_DEFAULT =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  private static boolean hasTag(String[] tags, String needle) {
    if (tags == null) return false;
    for (String t : tags) {
      if (needle.equals(t)) return true;
    }
    return false;
  }

  public static void main(String[] args) throws Exception {
    String mph = (args != null && args.length > 0 && args[0] != null && !args[0].isEmpty())
        ? args[0]
        : MPH_DEFAULT;

    Model model;
    try {
      model = ModelUtil.load("Model", mph);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + mph, e);
    }

    String exportTag = "";
    if (hasTag(model.result().export().tags(), "img_std1")) {
      exportTag = "img_std1";
    } else if (hasTag(model.result().export().tags(), "img_hr_pg_holo_std1")) {
      exportTag = "img_hr_pg_holo_std1";
    } else {
      throw new RuntimeException("No std1 export tag found (expected img_hr_pg_holo_std1 or img_std1).");
    }

    String out;
    try {
      out = model.result().export(exportTag).getString("pngfilename");
    } catch (Exception e) {
      out = "";
    }
    if (out == null || out.isEmpty()) {
      throw new RuntimeException("Export tag " + exportTag + " has no pngfilename configured.");
    }

    File outFile = new File(out);
    model.result().export(exportTag).run();

    long size = outFile.exists() ? outFile.length() : 0L;
    System.out.println(
        "STD1_EXPORT|model=" + mph + "|tag=" + exportTag + "|file=" + out + "|exists=" + outFile.exists() + "|size=" + size);
  }
}
