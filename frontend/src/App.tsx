import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import AnalyzePage from './pages/AnalyzePage';
import ResultPage from './pages/ResultPage';
import PerformancePage from './pages/PerformancePage';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<AnalyzePage />} />
          <Route path="/result/:id" element={<ResultPage />} />
          <Route path="/performance" element={<PerformancePage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
