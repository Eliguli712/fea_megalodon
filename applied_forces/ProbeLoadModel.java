import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeLoadModel {
  private static final String DEFAULT_MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics_visual_access_test.mph";

  public static void main(String[] args) throws Exception {
    String mph = (args != null && args.length == 1) ? args[0] : DEFAULT_MPH;
    Model model;
    try {
      model = ModelUtil.load("Model", mph);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load " + mph, e);
    }
    System.out.println("LOAD_OK|" + mph);
    try {
      System.out.println("STUDIES|" + String.join(",", model.study().tags()));
    } catch (Exception e) {
      System.out.println("STUDIES|<unavailable>");
    }
    try {
      System.out.println("PLOTS|" + String.join(",", model.result().tags()));
    } catch (Exception e) {
      System.out.println("PLOTS|<unavailable>");
    }
  }
}
