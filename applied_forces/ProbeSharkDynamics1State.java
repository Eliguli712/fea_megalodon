import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;

public class ProbeSharkDynamics1State {
  private static final String[] FILES = new String[] {
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics_1.mph",
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/shark_dynamics_1.mph"
  };

  private static void printList(String label, String[] values) {
    System.out.println(label + "|" + String.join(",", values));
  }

  public static void main(String[] args) throws Exception {
    for (String file : FILES) {
      Model model;
      try {
        model = ModelUtil.load("Model", file);
      } catch (IOException e) {
        throw new RuntimeException("Failed to load " + file, e);
      }

      System.out.println("FILE|" + file);

      try {
        printList("STUDIES", model.study().tags());
      } catch (Exception e) {
        System.out.println("STUDIES|<none>");
      }

      try {
        printList("DATASETS", model.result().dataset().tags());
      } catch (Exception e) {
        System.out.println("DATASETS|<none>");
      }

      try {
        printList("PLOT_GROUPS", model.result().tags());
      } catch (Exception e) {
        System.out.println("PLOT_GROUPS|<none>");
      }
    }
  }
}
