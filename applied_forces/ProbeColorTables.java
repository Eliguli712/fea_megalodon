import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeColorTables {
  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    String[] candidates = new String[]{
      "Rainbow", "RainbowLight", "Spectrum", "SpectrumLight", "Prism", "Traffic", "Thermal", "ThermalLight", "Wave", "Viridis", "Turbo", "Jet"
    };

    try { m.result().remove("pg_colprobe"); } catch (Exception e) {}
    m.result().create("pg_colprobe", "PlotGroup3D");
    m.result("pg_colprobe").set("data", "dset6");
    m.result("pg_colprobe").create("surf1", "Surface");
    m.result("pg_colprobe").feature("surf1").set("expr", "solid.mises");

    for (String c : candidates) {
      try {
        m.result("pg_colprobe").feature("surf1").set("colortable", c);
        System.out.println("OK " + c);
      } catch (Exception e) {
        System.out.println("BAD " + c + " : " + e.getMessage());
      }
    }
  }
}
