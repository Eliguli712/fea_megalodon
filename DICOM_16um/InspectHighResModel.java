import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Arrays;

public class InspectHighResModel {
  private static final String MODEL_PATH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics_high_resolution.mph";

  private static void printTags(String title, String[] tags) {
    System.out.println("== " + title + " ==");
    for (String tag : tags) {
      System.out.println(" - " + tag);
    }
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MODEL_PATH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load: " + MODEL_PATH, e);
    }

    printTags("component tags", model.component().tags());
    for (String compTag : model.component().tags()) {
      printTags(compTag + " geom tags", model.component(compTag).geom().tags());
      printTags(compTag + " mesh tags", model.component(compTag).mesh().tags());
      printTags(compTag + " physics tags", model.component(compTag).physics().tags());
      for (String meshTag : model.component(compTag).mesh().tags()) {
        printTags(compTag + "/" + meshTag + " feature tags", model.component(compTag).mesh(meshTag).feature().tags());
      }
    }
    printTags("global mesh tags", model.mesh().tags());
    if (Arrays.asList(model.mesh().tags()).contains("mpart1")) {
      printTags("mpart1 features", model.mesh("mpart1").feature().tags());
    }
    printTags("study tags", model.study().tags());
    for (String stdTag : model.study().tags()) {
      printTags(stdTag + " features", model.study(stdTag).feature().tags());
    }
    printTags("dataset tags", model.result().dataset().tags());
    printTags("result tags", model.result().tags());
  }
}
