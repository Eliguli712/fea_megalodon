import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class VerifyStaticDynamicsVonMises {
  private static final String MODEL_PATH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics.mph";

  private static final String[] REQUIRED_TAGS = new String[]{
      "pg_vm_point_smoothed",
      "pg_vm_surface_smoothed",
      "pg_vm_point_uncompressed",
      "pg_vm_surface_uncompressed",
      "pg_vm_point_rawtet",
      "pg_vm_surface_rawtet",
      "pg_vm_point_img_smoothed",
      "pg_vm_surface_img_smoothed",
      "pg_vm_point_img_uncompressed",
      "pg_vm_surface_img_uncompressed",
      "pg_vm_point_img_rawtet",
      "pg_vm_surface_img_rawtet"
  };

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MODEL_PATH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MODEL_PATH, e);
    }

    Set<String> existing = new HashSet<>(Arrays.asList(model.result().tags()));
    boolean allPresent = true;
    boolean allRunnable = true;
    boolean allImageResolvable = true;

    for (String tag : REQUIRED_TAGS) {
      boolean present = existing.contains(tag);
      if (!present) {
        allPresent = false;
        System.out.println("VERIFY_TAG|" + tag + "|present=false");
        continue;
      }
      System.out.println("VERIFY_TAG|" + tag + "|present=true");

      try {
        model.result(tag).run();
        System.out.println("VERIFY_RUN|" + tag + "|ok=true");
      } catch (Exception e) {
        allRunnable = false;
        System.out.println("VERIFY_RUN|" + tag + "|ok=false|" + e.getMessage());
      }

      try {
        String data = model.result(tag).getString("data");
        System.out.println("VERIFY_DATASET|" + tag + "|data=" + data);
      } catch (Exception ignored) {
      }

      if (tag.contains("_img_")) {
        try {
          String image = model.result(tag).feature("img1").getString("filename");
          File f = new File(image);
          boolean exists = f.exists();
          long size = exists ? f.length() : -1L;
          if (!exists || size <= 0) {
            allImageResolvable = false;
          }
          System.out.println(
              "VERIFY_IMAGE|" + tag + "|exists=" + exists + "|size=" + size + "|path=" + image
          );
        } catch (Exception e) {
          allImageResolvable = false;
          System.out.println("VERIFY_IMAGE|" + tag + "|exists=false|error=" + e.getMessage());
        }
      }
    }

    System.out.println("VERIFY_SUMMARY|all_present=" + allPresent
        + "|all_runnable=" + allRunnable
        + "|all_image_resolvable=" + allImageResolvable);
  }
}
