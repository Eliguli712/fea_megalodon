import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;

public class RunStrict3BdfImgStd1Only {
  private static final String MPH =
      "DICOM_16um/exports/static_dynamics_high_resolution_strict3bdf.mph";

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    if (model.result().export() == null) {
      throw new RuntimeException("No export features found.");
    }
    if (!hasTag(model.result().export().tags(), "img_std1")) {
      throw new RuntimeException("Export tag img_std1 not found.");
    }

    String out = "";
    try {
      out = model.result().export("img_std1").getString("pngfilename");
    } catch (Exception ignored) {
    }
    model.result().export("img_std1").run();
    System.out.println("IMG_STD1_EXPORT|file=" + out);
  }

  private static boolean hasTag(String[] tags, String needle) {
    if (tags == null) return false;
    for (String t : tags) {
      if (needle.equals(t)) return true;
    }
    return false;
  }
}
