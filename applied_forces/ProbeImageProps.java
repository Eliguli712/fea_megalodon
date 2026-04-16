import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeImageProps {
  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
    try { m.result().remove("pgi"); } catch (Exception e) {}
    m.result().create("pgi", "PlotGroup2D");
    m.result("pgi").create("img1", "Image");
    ResultFeature f = m.result("pgi").feature("img1");
    System.out.println("TYPE=" + f.getType());
    for (String p : f.properties()) {
      System.out.println(p + " type=" + f.getValueType(p));
    }
    for (String key : new String[]{"filename","imagefilename","source","sourcetype","importeddatatype","resolution","width","height","positioning","xdata","data"}) {
      try {
        System.out.println("ALLOWED " + key + " -> " + java.util.Arrays.toString(f.getAllowedPropertyValues(key)));
      } catch (Exception e) {
        System.out.println("ALLOWED " + key + " -> <err> " + e.getMessage());
      }
      try {
        System.out.println("HAS " + key + " -> " + f.hasProperty(key));
      } catch (Exception e) {}
    }
  }
}
