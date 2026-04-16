import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Arrays;

public class InspectTemplateModel {
  private static final String TEMPLATE_MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";

  private static void printTags(String title, String[] tags) {
    System.out.println("== " + title + " ==");
    for (String tag : tags) {
      System.out.println(" - " + tag);
    }
  }

  public static void main(String[] args) {
    Model model;
    try {
      model = ModelUtil.load("Model", TEMPLATE_MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load template: " + TEMPLATE_MPH, e);
    }

    printTags("mesh tags", model.mesh().tags());
    if (Arrays.asList(model.mesh().tags()).contains("mpart1")) {
      printTags("mesh mpart1 features", model.mesh("mpart1").feature().tags());
    }
    if (Arrays.asList(model.component().tags()).contains("comp1")) {
      printTags("comp1 mesh tags", model.component("comp1").mesh().tags());
      if (Arrays.asList(model.component("comp1").mesh().tags()).contains("mesh1")) {
        printTags("comp1 mesh1 features", model.component("comp1").mesh("mesh1").feature().tags());
      }
      printTags("comp1 physics tags", model.component("comp1").physics().tags());
      if (Arrays.asList(model.component("comp1").physics().tags()).contains("solid")) {
        printTags(
            "comp1 solid features",
            model.component("comp1").physics("solid").feature().tags()
        );
      }
    }

    printTags("study tags", model.study().tags());
    if (Arrays.asList(model.study().tags()).contains("std_mr5")) {
      printTags("std_mr5 features", model.study("std_mr5").feature().tags());
    }

    try {
      printTags("result dataset tags", model.result().dataset().tags());
    } catch (Exception ignored) {
    }

    ModelUtil.disconnect();
  }
}
