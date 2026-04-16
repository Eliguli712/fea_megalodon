import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeImageAllowed {
  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
    try { m.result().remove("pgi"); } catch (Exception e) {}
    m.result().create("pgi", "PlotGroup2D");
    m.result("pgi").create("img1", "Image");
    ResultFeature f = m.result("pgi").feature("img1");
    for (String key : new String[]{"mapping","coordinterpretation","anchorpos","heightmode","displacement","planetype","sourcetype"}) {
      try {
        System.out.println(key + " -> " + java.util.Arrays.toString(f.getAllowedPropertyValues(key)));
      } catch (Exception e) {
        System.out.println(key + " err " + e.getMessage());
      }
    }
  }
}
