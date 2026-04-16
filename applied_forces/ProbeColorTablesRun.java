import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeColorTablesRun {
  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    String[] candidates = new String[]{"RainbowLight","Rainbow","SpectrumLight","Spectrum","Traffic","Thermal"};
    for (String c : candidates) {
      String pg = "pgc_" + c.toLowerCase().replace(" ","_");
      try { m.result().remove(pg); } catch (Exception e) {}
      try {
        m.result().create(pg, "PlotGroup3D");
        m.result(pg).set("data", "dset6");
        m.result(pg).create("surf1", "Surface");
        m.result(pg).feature("surf1").set("expr", "solid.mises");
        m.result(pg).feature("surf1").set("colortable", c);
        m.result(pg).run();
        System.out.println("OKRUN " + c);
      } catch (Exception e) {
        System.out.println("BADRUN " + c + " :: " + e.getMessage());
      }
    }
  }
}
