import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Arrays;

public class ProbePointProps {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics.mph";

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    try {
      model.result().remove("pg_probe_point_props");
    } catch (Exception ignored) {
    }
    model.result().create("pg_probe_point_props", "PlotGroup3D");
    model.result("pg_probe_point_props").create("pt1", "Point");
    ResultFeature pt = model.result("pg_probe_point_props").feature("pt1");

    String[] props = pt.properties();
    Arrays.sort(props);
    System.out.println("POINT_PROP_COUNT=" + props.length);
    for (String key : props) {
      try {
        String type = pt.getValueType(key);
        String[] allowed = pt.getAllowedPropertyValues(key);
        System.out.println("POINT_PROP key=" + key + " type=" + type + " allowed=" + (allowed == null ? "null" : Arrays.toString(allowed)));
      } catch (Exception e) {
        System.out.println("POINT_PROP key=" + key + " err=" + e.getMessage());
      }
    }
  }
}
