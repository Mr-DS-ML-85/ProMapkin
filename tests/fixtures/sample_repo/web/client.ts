/** Browser client for the search API. */
import axios from "axios";

export interface SearchResult { id: string; text: string }

export class SearchClient {
  constructor(private baseUrl: string) {}

  async search(query: string): Promise<SearchResult[]> {
    const response = await axios.get(`${this.baseUrl}/search`, { params: { q: query } });
    return formatResults(response.data.results);
  }
}

export function formatResults(results: SearchResult[]): SearchResult[] {
  return results.filter((r) => r.text.length > 0);
}
